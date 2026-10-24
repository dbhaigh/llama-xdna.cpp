from pathlib import Path
import os

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.config import cxx_header_path

K = int(os.environ.get("GGML_XDNA_Q4_0_K", "256"))
M = int(os.environ.get("GGML_XDNA_Q4_0_M", "16"))
TILE_K = 256
TILE_M = 16
BLOCK_BYTES = 18
BLOCK_SIZE = 32
TILE_K_BYTES = TILE_K // BLOCK_SIZE * BLOCK_BYTES
TILE_WEIGHT_BYTES = TILE_M * TILE_K_BYTES
M_TILES = M // TILE_M
K_TILES = K // TILE_K
WEIGHTS_BYTES = M * (K // BLOCK_SIZE) * BLOCK_BYTES
BATCH_M_TILES = 64

if K <= 0 or K % TILE_K != 0 or M <= 0 or M % TILE_M != 0:
    raise ValueError("Q4_0 matvec dimensions must be positive multiples of 256 (K) and 16 (M)")

weights_type = np.ndarray[(WEIGHTS_BYTES,), np.dtype[np.uint8]]
weights_tile_type = np.ndarray[(TILE_WEIGHT_BYTES,), np.dtype[np.uint8]]
input_type = np.ndarray[(K,), np.dtype[np.float32]]
output_type = np.ndarray[(M,), np.dtype[np.float32]]


@iron.jit(use_cache=False)
def q4_0_matvec(weights: In, input: In, output: Out):
    weights_fifo = ObjectFifo(weights_tile_type, depth=1, name="weights")
    input_fifo = ObjectFifo(input_type, depth=1, name="input")
    output_tile_type = np.ndarray[(TILE_M,), np.dtype[np.float32]]
    output_fifo = ObjectFifo(output_tile_type, depth=1, name="output")

    kernel = ExternalFunction(
        "q4_0_matvec",
        source_file=str(Path(__file__).with_name("q4_0_matvec.cc")),
        arg_types=[weights_tile_type, input_type, output_tile_type, np.int32],
        include_dirs=[cxx_header_path()],
    )

    def core_body(weights_fifo, input_fifo, output_fifo, kernel):
        input_elem = input_fifo.acquire(1)
        for _ in range_(M_TILES):
            output_elem = output_fifo.acquire(1)
            for k_tile in range_(K_TILES):
                weights_elem = weights_fifo.acquire(1)
                kernel(weights_elem, input_elem, output_elem, k_tile)
                weights_fifo.release(1)
            output_fifo.release(1)
        input_fifo.release(1)

    worker = Worker(
        core_body,
        fn_args=[
            weights_fifo.cons(),
            input_fifo.cons(),
            output_fifo.prod(),
            kernel,
        ],
        dynamic_objfifo_lowering=True,
    )

    def sequence(
        weights,
        input,
        output,
        weights_prod,
        input_prod,
        output_cons,
    ):
        row_bytes = K * BLOCK_BYTES // BLOCK_SIZE
        input_prod.fill(input)
        for first_m_tile in range(0, M_TILES, BATCH_M_TILES):
            batch_m_tiles = min(BATCH_M_TILES, M_TILES - first_m_tile)
            weights_tap = TensorAccessPattern(
                (WEIGHTS_BYTES,),
                first_m_tile * TILE_M * row_bytes,
                (batch_m_tiles, K_TILES, TILE_M, TILE_K_BYTES),
                (TILE_M * row_bytes, TILE_K_BYTES, row_bytes, 1),
            )
            output_tap = TensorAccessPattern(
                (M,),
                first_m_tile * TILE_M,
                (batch_m_tiles, TILE_M),
                (TILE_M, 1),
            )
            weights_task = weights_prod.fill(
                weights,
                tap=weights_tap,
                wait=True,
                managed=False,
            )
            output_task = output_cons.drain(
                output,
                tap=output_tap,
                wait=True,
                managed=False,
            )
            output_task.await_()
            weights_task.await_()

    runtime = Runtime(
        sequence,
        [
            weights_type,
            input_type,
            output_type,
            weights_fifo.prod(),
            input_fifo.prod(),
            output_fifo.cons(),
        ],
    )

    program = Program(iron.get_current_device(), runtime, workers=[worker])
    return program.resolve_program()


if __name__ == "__main__":
    output_dir = Path(os.environ.get("GGML_XDNA_KERNEL_OUTPUT_DIR", Path(__file__).resolve().parent))
    q4_0_matvec.compile(
        xclbin_path=output_dir / "q4_0_matvec.xclbin",
        inst_path=output_dir / "q4_0_matvec.insts.bin",
        pdi_path=output_dir / "q4_0_matvec.pdi",
    )
