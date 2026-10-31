"""Process-local evaluation compatibility fixes; original libraries stay intact."""
import os
import random
import sys


def preserve_gpqa_input(self, record):
    """Original GPQA per-record shuffle, preserving scientific square brackets."""
    def preprocess(text):
        if text is None:
            return ' '
        return text.strip().replace(' [title]', '. ').replace('  ', ' ')

    choices = [preprocess(record[key]) for key in
               ('Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3', 'Correct Answer')]
    correct = preprocess(record['Correct Answer'])
    if choices.count(correct) != 1:
        raise ValueError('GPQA-Diamond correct answer collides with a distractor')
    random.shuffle(choices)
    return {'choices': choices, 'answer': chr(65 + choices.index(correct))}


preserve_gpqa_input.preserves_square_brackets = True


def install_gpqa_choice_preservation():
    from evalscope.benchmarks.gpqa.gpqa_adapter import GPQAAdapter
    GPQAAdapter._process_input = preserve_gpqa_input


def _install_dense_dp_load_balancing():
    import functools
    import importlib.metadata
    import os
    from collections import Counter

    mode = os.environ.get('VLLM_DENSE_DP_LB_FIX', 'auto').strip().lower()
    if mode not in {'auto', '0', '1'}:
        raise ValueError('VLLM_DENSE_DP_LB_FIX must be auto, 0, or 1')
    if mode == '0':
        return {'installed': False, 'reason': 'disabled'}

    version = importlib.metadata.version('vllm')
    if mode == 'auto' and version.split('+', 1)[0] != '0.23.0':
        return {'installed': False, 'reason': 'unverified vLLM version; using native routing',
                'vllm_version': version}

    from vllm.v1.engine.core_client import DPLBAsyncMPClient

    patch_id = 'gpqa_dense_dp_least_inflight_v1'
    if getattr(DPLBAsyncMPClient, '_gpqa_dense_dp_lb_patch_id', None) == patch_id:
        return {'installed': True, 'reason': 'already installed', 'vllm_version': version}
    if getattr(DPLBAsyncMPClient, '_gpqa_dense_dp_lb_patch_id', None) is not None:
        raise RuntimeError('A different GPQA DP routing patch is already installed')
    for method in ('get_core_engine_for_request', 'process_engine_outputs', 'abort_requests_async'):
        if not callable(getattr(DPLBAsyncMPClient, method, None)):
            raise RuntimeError(f'Unsupported vLLM DP client API: missing {method}')

    native_init = DPLBAsyncMPClient.__init__
    native_choose = DPLBAsyncMPClient.get_core_engine_for_request

    @functools.wraps(native_init)
    def initialize(self, vllm_config, *args, **kwargs):
        native_init(self, vllm_config, *args, **kwargs)
        parallel = vllm_config.parallel_config
        # This script deploys one front end and independent dense TP=1 replicas.
        # Keep all other deployment modes on their original routing path.
        self._gpqa_dense_dp_lb_enabled = (
            not getattr(vllm_config.model_config, 'is_moe', True)
            and getattr(self, 'client_count', None) == 1
            and getattr(parallel, 'tensor_parallel_size', None) == 1
            and not getattr(parallel, 'data_parallel_external_lb', False)
            and not getattr(parallel, 'data_parallel_hybrid_lb', False)
            and not getattr(parallel, 'enable_elastic_ep', False)
            and len(self.core_engines) > 1
        )
        self._gpqa_dense_dp_lb_next = getattr(self, 'eng_start_index', 0) % len(self.core_engines)
        self._gpqa_dense_dp_lb_assigned = [0] * len(self.core_engines)
        if self._gpqa_dense_dp_lb_enabled:
            if not isinstance(getattr(self, 'reqs_in_flight', None), dict):
                raise RuntimeError('Unsupported vLLM DP client: reqs_in_flight is not a dict')
            print(f'[info] Evaluation dense-DP load balancing enabled: '
                  f'{len(self.core_engines)} replicas, least in-flight, rotating ties', flush=True)
        else:
            print('[info] Evaluation dense-DP workaround not applicable; using native routing', flush=True)

    @functools.wraps(native_choose)
    def choose(self, request):
        if (not getattr(self, '_gpqa_dense_dp_lb_enabled', False)
                or request.data_parallel_rank is not None
                or request.pooling_params is not None):
            return native_choose(self, request)

        previous = self.reqs_in_flight.get(request.request_id)
        if previous is not None:
            return previous

        # This synchronous selector and the native completion callback run on
        # the same event loop. No await occurs between counting and assigning.
        # Coordinator snapshots can reset lb_engines, but never this request map.
        counts = Counter(self.reqs_in_flight.values())
        engines = self.core_engines
        start = self._gpqa_dense_dp_lb_next
        rank = min(((start + i) % len(engines) for i in range(len(engines))),
                   key=lambda i: counts[engines[i]])
        engine = engines[rank]
        self.reqs_in_flight[request.request_id] = engine
        self._gpqa_dense_dp_lb_next = (rank + 1) % len(engines)
        self._gpqa_dense_dp_lb_assigned[rank] += 1

        assigned = self._gpqa_dense_dp_lb_assigned
        total = sum(assigned)
        if total <= len(engines) or total % 64 == 0:
            outstanding = [counts[engine_id] + (engine_id == engine) for engine_id in engines]
            print(f'[info] Evaluation dense-DP routes: assigned={assigned}, '
                  f'in_flight={outstanding}, selected_rank={rank}', flush=True)
        # Preserve native process_engine_outputs and abort_requests_async.
        # They remove the mapping only when the engine reports completion,
        # including an acknowledged abort; do not release it prematurely.
        return engine

    DPLBAsyncMPClient.__init__ = initialize
    DPLBAsyncMPClient.get_core_engine_for_request = choose
    DPLBAsyncMPClient._gpqa_dense_dp_lb_patch_id = patch_id
    return {'installed': True, 'strategy': 'least_inflight_with_rotating_ties',
            'vllm_version': version, 'mode': mode}


if os.environ.get('LLAMA_EVAL_STAGE', '').strip().lower() == 'gpqa':
    install_gpqa_choice_preservation()
    print('[info] GPQA-Diamond: preserving square brackets in answer choices', file=sys.stderr, flush=True)

DENSE_DP_LB_PATCH_STATUS = _install_dense_dp_load_balancing()
print(f'[info] Evaluation dense-DP patch: {DENSE_DP_LB_PATCH_STATUS}', file=sys.stderr, flush=True)
