"""Réduit la taille d'un modèle ONNX : poids stockés en int8 (par canal de
sortie, symétrique) ou en float16, re-convertis en float32 au chargement par
un nœud DequantizeLinear ou Cast. Les calculs restent en float32.

    python compress_onnx.py entree.onnx sortie.onnx int8|fp16
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def main(src, dst, mode):
    m = onnx.load(src)
    g = m.graph
    new_inits, new_nodes = [], []
    for init in list(g.initializer):
        w = numpy_helper.to_array(init)
        if w.dtype != np.float32 or w.size < 1024:
            new_inits.append(init)
            continue
        name = init.name
        if mode == "fp16":
            new_inits.append(numpy_helper.from_array(w.astype(np.float16), name + "_fp16"))
            new_nodes.append(helper.make_node("Cast", [name + "_fp16"], [name], to=TensorProto.FLOAT))
        else:
            flat = w.reshape(w.shape[0], -1)
            scale = np.abs(flat).max(axis=1) / 127.0
            scale[scale == 0] = 1.0
            q = np.clip(np.round(flat / scale[:, None]), -127, 127).astype(np.int8).reshape(w.shape)
            new_inits.append(numpy_helper.from_array(q, name + "_q"))
            new_inits.append(numpy_helper.from_array(scale.astype(np.float32), name + "_s"))
            new_nodes.append(helper.make_node("DequantizeLinear", [name + "_q", name + "_s"], [name], axis=0))
    del g.initializer[:]
    g.initializer.extend(new_inits)
    nodes = new_nodes + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, dst)


if __name__ == "__main__":
    main(*sys.argv[1:4])
