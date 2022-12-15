#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>

#include "fft.h"

#include <random>
#include <limits>
#include <omp.h>

// Built with: dpcpp FFT_c2c_backward.cpp -DMKL_ILP64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -L$HOME/fftpack -lfftpack

int nx = 32;
int nq1 = 544;
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

// Original implementation as seen in PWDFT/Nwpw/nwpwlib/D3db/d3db.cpp
void fft_c2c_fftpack(double* a) {
    double* tmpx = new double[2*(2*nx+15)];
    dcffti_(&nx, tmpx);

    int indx = 0;
    for (int q=0; q<nq1; ++q) {
	dcfftb_(&nx, &a[indx], tmpx);
	indx += (nx*2);
    }

    return;
}


// using non-batch version of oneMKL FFT
void fft_c2c_nonbatch_onemkl(double* a) {
    sycl::queue dev_que(sycl::gpu_selector{},
			    sycl::property_list{sycl::property::queue::in_order{}});

    double* x_usm = sycl::malloc_device<double>(n2ft3d, dev_que);

    //copy input from host to device
    auto h2d = dev_que.memcpy(x_usm, a, n2ft3d*sizeof(double));
    h2d.wait();

    typedef oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
					 oneapi::mkl::dft::domain::COMPLEX> descriptor_t;
    descriptor_t desc(nx);
    desc.commit(dev_que);

    int indx = 0;
    for (int q=0; q<nq1; ++q) {
	auto fft = compute_backward(desc, &(x_usm[indx]));
	indx += (nx*2);

	fft.wait();
    }

    //copy input from device to host
    auto d2h = dev_que.memcpy(a, x_usm, n2ft3d*sizeof(double));
    d2h.wait();

    sycl::free(x_usm, dev_que);
}

// using batch version of oneMKL FFT
void fft_c2c_batch_onemkl(double* a) {
    sycl::queue dev_que(sycl::gpu_selector{},
			    sycl::property_list{sycl::property::queue::in_order{}});

    double* x_usm = sycl::malloc_device<double>(n2ft3d, dev_que);

    //copy input from host to device
    auto h2d = dev_que.memcpy(x_usm, a, n2ft3d*sizeof(double));
    h2d.wait();

    typedef oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
					 oneapi::mkl::dft::domain::COMPLEX> descriptor_t;
    descriptor_t desc(nx);
    desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, nq1);
    desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, nx);
    desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, nx);
    desc.commit(dev_que);

    auto fft = compute_backward(desc, x_usm);
    fft.wait();

    //copy output from device to host
    auto d2h = dev_que.memcpy(a, x_usm, n2ft3d*sizeof(double));
    d2h.wait();

    sycl::free(x_usm, dev_que);
}


int main(int argc, char** argv) {
    try {

         if (argc != 4) throw 42;
         sscanf(argv[1], "%d", &nx);
         sscanf(argv[2], "%d", &nq1);
         sscanf(argv[3], "%d", &n2ft3d);
         std::cout << "Test nx , nq1, n2ft3d = " << nx << ","<< nq1 << ","<< n2ft3d << std::endl;
    }
        catch ( int err)
        {
        std::cout << "Use as: ./program <int nb> <int nc>" <<std::endl;
        std::cout << "example : ./program 32 544 34816" <<std::endl;
                return -1;
        }


    double* inout_fftpack = new double[n2ft3d];
    double* inout_nonbatch_onemkl = new double[n2ft3d];
    double* inout_batch_onemkl = new double[n2ft3d];

    init(inout_fftpack, n2ft3d/2);
    memcpy(inout_nonbatch_onemkl, inout_fftpack, n2ft3d*sizeof(double));
    memcpy(inout_batch_onemkl, inout_fftpack, n2ft3d*sizeof(double));

    const double itime = omp_get_wtime();

    fft_c2c_fftpack(inout_fftpack);
    fft_c2c_nonbatch_onemkl(inout_nonbatch_onemkl);
    fft_c2c_batch_onemkl(inout_batch_onemkl);

    for(int k=0; k<n2ft3d; k++) {
    	if( std::abs(inout_fftpack[k] - inout_batch_onemkl[k]) > 1.0e-12 ) {
    	    std::cout << " output[" << k << "] : " << inout_fftpack[k] << ", " << inout_nonbatch_onemkl[k] << ", " << inout_batch_onemkl[k] << std::endl;
    	}
    }

    const  double ftime = omp_get_wtime();
    const double exec_time = ftime - itime;

    std::cout << "Ok" << std::endl;
    std::cout << "Time =" << exec_time  << std::endl;

    delete[] inout_fftpack;
    delete[] inout_nonbatch_onemkl;
    delete[] inout_batch_onemkl;

    return 0;
}
