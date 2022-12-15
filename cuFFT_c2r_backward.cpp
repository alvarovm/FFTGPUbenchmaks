#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

#include "fft.h"

#include <sstream>
#include <iostream>
#include <cstring>
#include <random>
#include <limits>
#include <omp.h>

// Built with: nvcc cuFFT_c2r_backward.cpp -L$HOME/fftpack_gnu -lfftpack -lcufft -o cufft_c2r_backward

//int nx = 33;
//int nq1 = 1;
//int n2ft3d = 35;

int nx = 32;
int nq1 = 1024;
int n2ft3d = 34816;

static void init(double *x, int N)
{
    std::random_device rd;
    std::default_random_engine generator(rd()); // rd() provides a random seed
    std::uniform_real_distribution<double> distribution(0.1, 1);

    for (int n = 0; n < N; ++n) {
        x[2*n+0] = distribution(generator) / 1000.0;
        x[2*n+1] = distribution(generator) / 1000.0;
    }
}

static void cshift1_fftb(const int n1,const int n2, const int n3, const int n4, double *a)
{
    int i,j,indx;
    indx = 1;
    for (j=0; j<(n2*n3*n4); ++j)
    {
	for (i=2; i<=n1; ++i)
	{
	    a[indx+i-2] = a[indx+i-1];
	}
	indx += (n1+2);
    }
}

// Original implementation as seen in PWDFT/Nwpw/nwpwlib/D3db/d3db.cpp
void fft_c2r_fftpack(double* a) {
    double* tmpx = new double[2*(2*nx+15)];
    drffti_(&nx, tmpx);

    cshift1_fftb(nx,nq1,1,1,a);
    int indx = 0;
    for (int q=0; q<nq1; ++q) {
	drfftb_(&nx, &a[indx], tmpx);
	indx += (nx+2);
    }

    return;
}

// using non-batch version of cufft FFT
void fft_c2r_nonbatch_cufft(double* a) {
    double* x_usm = NULL;
    cudaMalloc((void**)&x_usm, n2ft3d * sizeof(double));

    //copy input from host to device
    cudaMemcpy(x_usm, a, n2ft3d*sizeof(double), cudaMemcpyHostToDevice);

    cufftHandle plan=0;

    if (cufftPlan1d(&plan, nx, CUFFT_Z2D, 1) != CUFFT_SUCCESS){
    	std::stringstream msg;
    	msg << "cuFFT (non-batch) Plan Error at " << __FILE__ << " : " << __LINE__ << ", " << cudaPeekAtLastError() << std::endl;
    	return;
    }

    int indx = 0;
    for (int q=0; q<nq1; ++q) {
	if (cufftExecZ2D(plan,
			 reinterpret_cast<cufftDoubleComplex *>(&x_usm[indx]),
			 reinterpret_cast<cufftDoubleReal *>(&x_usm[indx])) != CUFFT_SUCCESS) {
	    std::stringstream msg;
	    msg << "cuFFT (non-batch) Exec Error at " << __FILE__ << " : " << __LINE__ << ", " << cudaPeekAtLastError() << std::endl;
	    return;
	}
	indx += (nx+2);
    }

    //copy input from device to host
    cudaMemcpy(a, x_usm, n2ft3d*sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    
    cufftDestroy(plan);
    cudaFree(x_usm);
}

// using batch version of cufft FFT
void fft_c2r_batch_cufft(double* a) {

    double* x_usm = NULL;
    cudaMalloc((void**)&x_usm, n2ft3d * sizeof(double));

    //copy input from host to device
    cudaMemcpy(x_usm, a, n2ft3d*sizeof(double), cudaMemcpyHostToDevice);

    cufftHandle plan=0;

    if (cufftPlan1d(&plan, nx, CUFFT_Z2D, nq1) != CUFFT_SUCCESS){
    	std::stringstream msg;
    	msg << "cuFFT Exec Error at " << __FILE__ << " : " << __LINE__ << ", " << cudaPeekAtLastError() << std::endl;
    	return;
    }

    // int inembed[] = {nx};
    // int onembed[] = {nx};
    // int idist = nx;
    // int odist = nx+2;
    // if (cufftPlanMany(&plan, 1, &nx,
    // 		      inembed, 1, idist,
    // 		      onembed, 1, odist,
    // 		      CUFFT_Z2D, nq1) != CUFFT_SUCCESS) {
    // 	std::stringstream msg;
    // 	msg << "cuFFT Plan Error at " << __FILE__ << " : " << __LINE__ << ", " << cudaPeekAtLastError() << std::endl;
    // 	return;
    // }

    if (cufftExecZ2D(plan,
		     reinterpret_cast<cufftDoubleComplex*>(x_usm),
		     reinterpret_cast<cufftDoubleReal*>(x_usm)) != CUFFT_SUCCESS) {
	std::stringstream msg;
	msg << "cuFFT Exec Error at " << __FILE__ << " : " << __LINE__ << ", " << cudaPeekAtLastError() << std::endl;
	return;
    }

    //copy input from device to host
    cudaMemcpy(a, x_usm, n2ft3d*sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cufftDestroy(plan);
    cudaFree(x_usm);
}

int main() {

    double* inout_fftpack = new double[n2ft3d];
    double* inout_nonbatch_cufft = new double[n2ft3d];
    double* inout_batch_cufft = new double[n2ft3d];

    init(inout_fftpack, n2ft3d/2);
    memcpy(inout_nonbatch_cufft, inout_fftpack, n2ft3d*sizeof(double));
    memcpy(inout_batch_cufft, inout_fftpack, n2ft3d*sizeof(double));

    const double itime = omp_get_wtime();

    fft_c2r_fftpack(inout_fftpack);
    fft_c2r_nonbatch_cufft(inout_nonbatch_cufft);
    fft_c2r_batch_cufft(inout_batch_cufft);

    for(int k=0; k<n2ft3d; k++) {
    	if( (std::abs(inout_fftpack[k] - inout_batch_cufft[k]) > 1.0e-12) ||
	    (std::abs(inout_fftpack[k] - inout_nonbatch_cufft[k]) > 1.0e-12) ) {
	    std::cout << " output[" << k << "] : " << inout_fftpack[k] << ", " << inout_nonbatch_cufft[k] << ", " << inout_batch_cufft[k] << std::endl;
	}
    }

    const  double ftime = omp_get_wtime();
    const double exec_time = ftime - itime;

    std::cout << "Ok" << std::endl;
    std::cout << "Time =" << exec_time  << std::endl;

    delete[] inout_fftpack;
    delete[] inout_nonbatch_cufft;
    delete[] inout_batch_cufft;

    return 0;
}
