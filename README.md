NCF: Neural Collapse Feature for OOD Detection (OpenOOD Plugin)
OpenOOD Post-Hoc Training-Free Easy-Integration

NCF (Neural Collapse Feature) is a post-hoc, training-free Out-of-Distribution (OOD) detection method integrated into OpenOOD.

It exploits the Neural Collapse phenomenon of well-converged classifiers to produce a strong OOD score with only one forward pass per test sample and one epoch of ID feature extraction for setup.

✨ Highlights
🚀 Plug-and-play — works with any OpenOOD backbone that exposes pre-logits features (resnet18_32x32, resnet50, vit-b-16, ...).
🔧 Training-free — no extra training step, no extra OOD data.
🎛 Single hyper-parameter — l1_lambda controls the feature-magnitude penalty; default sweep [0.1, 0.01, 0.001].
📦 First-class OpenOOD support — APS sweep, evaluator, sweep script, and ImageNet evaluator all work out of the box.
📖 Method at a glance
For a test feature z with predicted class ŷ = argmax(W z + b) and the class-mean prototype μ_{ŷ} (computed on the ID training set):

score(z) = cohesion(z, μ_{ŷ}) + l1_lambda * ||z||_1
where

cohesion(z, μ) = cos(z, μ) (default) or -||z - μ||_2,
l1_lambda is a single non-negative scalar.
Higher score ⇒ more likely in-distribution. The full mathematical intuition and references are documented in docs/postprocessors/ncf.md.

🛠 Installation
NCF is shipped with OpenOOD itself. Install OpenOOD as usual:

pip install git+https://github.com/Jingkang50/OpenOOD
pip install libmr
If you already have OpenOOD installed and only want the NCF files, copy:

openood/postprocessors/ncf_postprocessor.py
configs/postprocessors/ncf.yml
docs/postprocessors/ncf.md
scripts/ood/ncf/*.sh
scripts/sweep/sweep_posthoc_ncf.sh
into your OpenOOD working tree, then register the postprocessor by adding the following to openood/postprocessors/__init__.py and utils.py (this has already been done in the bundled release):

from .ncf_postprocessor import NCFPostprocessor
'ncf': NCFPostprocessor,  # inside get_postprocessor()
🚀 Quick start
1. Evaluate on CIFAR-10
python main.py \
  --config configs/datasets/cifar10/cifar10.yml \
           configs/datasets/cifar10/cifar10_ood.yml \
           configs/networks/resnet18_32x32.yml \
           configs/pipelines/test/test_ood.yml \
           configs/preprocessors/base_preprocessor.yml \
           configs/postprocessors/ncf.yml \
  --network.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'
…or simply:

sh scripts/ood/ncf/cifar10_test_ood_ncf.sh
2. Evaluate on ImageNet-1K (via unified evaluator)
python scripts/eval_ood_imagenet.py \
  --tvs-pretrained \
  --arch resnet50 \
  --postprocessor ncf \
  --save-score --save-csv
3. Run the full hyper-parameter sweep
sh scripts/sweep/sweep_posthoc_ncf.sh
This sweeps l1_lambda ∈ {0.1, 0.01, 0.001} × distance_type ∈ {cosine, euclidean} on CIFAR-10 and CIFAR-100.

⚙️ Configuration
configs/postprocessors/ncf.yml:

postprocessor:
  name: ncf
  APS_mode: True
  postprocessor_args:
    l1_lambda: 0.01
    distance_type: cosine
  postprocessor_sweep:
    l1_lambda_list: [0.1, 0.01, 0.001]
    distance_type_list: ['cosine', 'euclidean']
Key	Default	Description
l1_lambda	0.01	Weight of the feature-magnitude penalty term.
distance_type	'cosine'	'cosine' (similarity) or 'euclidean' (negative L2).
APS_mode	True	Enables OpenOOD's automatic hyper-parameter search.
To change the sweep grid, simply edit l1_lambda_list and/or distance_type_list in configs/postprocessors/ncf.yml.

📁 What this release adds to OpenOOD
Path	Purpose
openood/postprocessors/ncf_postprocessor.py	NCF postprocessor implementation.
configs/postprocessors/ncf.yml	Default config + sweep grid [0.1, 0.01, 0.001].
docs/postprocessors/ncf.md	Long-form documentation (background, math, file map).
scripts/ood/ncf/cifar10_test_ood_ncf.sh	CIFAR-10 evaluation script.
scripts/ood/ncf/cifar100_test_ood_ncf.sh	CIFAR-100 evaluation script.
scripts/ood/ncf/imagenet200_test_ood_ncf.sh	ImageNet-200 evaluation script.
scripts/ood/ncf/imagenet_test_ood_ncf.sh	ImageNet-1K evaluation script (via eval_ood_imagenet.py).
scripts/sweep/sweep_posthoc_ncf.sh	Full sweep over the NCF hyper-parameters.
openood/postprocessors/__init__.py (modified)	Registers NCFPostprocessor.
openood/postprocessors/utils.py (modified)	Maps 'ncf' → NCFPostprocessor in get_postprocessor.
README.md (modified)	Adds NCF to the supported-methods list.
🧪 Reproducibility
NCF is deterministic at inference time given the ID training features. For exact reproducibility:

Fix --seed 0 (already the OpenOOD default).
Use the checkpoints linked in the main OpenOOD README.
Report the sweep value selected by APS_mode (the *_best_* column in the generated metrics CSV).
🧠 References
Neural Collapse: V. Papyan, X. Y. Han, D. L. Donoho. Prevalence of neural collapse during the terminal phase of deep learning training. JMLR 2020.
OpenOOD v1.5: J. Zhang et al. OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection. arXiv:2306.09301, 2023.
OpenOOD repository: https://github.com/Jingkang50/OpenOOD
📝 License
This plugin follows the OpenOOD license (see LICENSE in the OpenOOD root).

🤝 Contributing
Issues and pull-requests should be opened against the upstream OpenOOD repository. Please prefix NCF-related PRs with [NCF].
