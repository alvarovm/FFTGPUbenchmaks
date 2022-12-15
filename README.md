# oneMKL vs CUDA FFT routines
---

Colletion of routines to compare FFT routines in oneMKL and cuFFT.


## Build instructions in JLSE
### Clone this repo

```
git clone  git@github.com:alvarovm/FFTGPUbenchmaks.git
```

### CUDA
```
qsub  -I -t 30  -n 1 -q gpu_v100_smx2_debug //login to a CUDA node
module purge
module add cuda gcc
cd FFTGPUbenchmaks
make -f makefile_cuda
```

### SYCL
```
qsub  -I -t 30  -n 1 -q iris //login to a SYCL node
module purge
module add oneapi/release/latest gcc
cd FFTGPUbenchmaks
make -f makefile_sycl
```


---

## Authors
* Abisheck Bagusetty (main) [@abagusetty](https://github.com/abagusetty)
* [@alvarovm](https://github.com/alvarovm) (editor) 

