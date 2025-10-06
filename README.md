# Curvy ILT

The source code of 

**Yang, Haoyu** and **Ren, Haoxing**.  
**"GPU-Accelerated Inverse Lithography Towards High Quality Curvy Mask Generation."**  
*Proceedings of the 2025 International Symposium on Physical Design*, pp. 42–50, 2025.


## Update

08/25/2025: We enabled batch optimization, in `run_iccad13_batch.py`, 10 ILT runs in parallel and finish in 4s on RTX 6000 ADA. Morph is disabled in forward_test because it cost lots of memory.

10/03/2025: Full chip ILT is enabled, checkout `fullchip` branch for details.

## Prepare

Install Package Dependencies

`pip install -r requirements.txt`

Install OpenCV seperately

`$sudo apt-get install python3-opencv`

## Usage

This is a running example on ICCAD13 benchmark images.
Change the directory for your own design.

`python3 run_iccad13.py`

We did not include the EPE computing code in this repo, please use [neuralILT](https://github.com/cuhk-eda/neural-ilt) or other 3rd party tools. 

## Contact

[Haoyu Yang](mailto:haoyuy@nvidia.com)





