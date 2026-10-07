"""llama.cpp の異種 GPU 投機的デコード（speculative decoding）ベンチ。

RX 6400 に本体モデル、WX 2100 にドラフトモデルを置く構成と、第 2 GPU を容量（layer 分割）に使う構成を、
条件を固定して比較する。詳細は docs/SPECBENCH.md。

  python -m tb250distill.specbench.run --plan main --out runs/specbench/<名前>
  python -m tb250distill.specbench.report runs/specbench/<名前>
"""
