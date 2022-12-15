#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>

#include "fft.h"

#include <random>
#include <limits>
#include <omp.h>

// Built with: dpcpp FFT_r2c_forward.cpp -DMKL_ILP64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -L$HOME/fftpack -lfftpack

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

static void cshift_fftf(const int n1,const int n2, const int n3, const int n4, double *a)
{
    int i,j,indx;
    indx = 1;
    for (j=0; j<(n2*n3*n4); ++j)
    {
	for (i=n1; i>=2; --i)
	{
	    a[indx+i-1] = a[indx+i-2];
	}
	a[indx+1-1]    = 0.0;
	a[indx+n1+1-1] = 0.0;
	indx += (n1+2);
    }
}

// Original implementation as seen in PWDFT/Nwpw/nwpwlib/D3db/d3db.cpp
void fft_r2c_fftpack(double* a) {
    double* tmpx = new double[2*(2*nx+15)];
    drffti_(&nx, tmpx);

    int indx = 0;
    for (int q=0; q<nq1; ++q) {
	drfftf_(&nx, &a[indx], tmpx);
	indx += (nx+2);
    }
    cshift_fftf(nx,nq1,1,1,a);

    return;
}

// using non-batch version of oneMKL FFT
void fft_r2c_nonbatch_onemkl(double* a) {
    sycl::queue dev_que(sycl::gpu_selector{},
			    sycl::property_list{sycl::property::queue::enable_profiling{},
				    sycl::property::queue::in_order{}});

    double* x_usm = sycl::malloc_device<double>(n2ft3d, dev_que);

    //copy input from host to device
    auto h2d = dev_que.memcpy(x_usm, a, n2ft3d*sizeof(double));
    h2d.wait();

    typedef oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
					 oneapi::mkl::dft::domain::REAL> descriptor_t;
    descriptor_t desc(nx);
    // Notes: looks from the outptut has the DFTI_CCS_FORMAT, need to check DFTI_PACK_FORMAT, DFTI_PERM_FORMAT
    desc.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    desc.set_value(oneapi::mkl::dft::config_param::PACKED_FORMAT, DFTI_CCS_FORMAT);
    desc.commit(dev_que);

    
    int indx = 0;
    // auto submission_time = 0.0;
    // auto execution_time = 0.0;   
    for (int q=0; q<nq1; ++q) {
	auto event = compute_forward(desc, &(x_usm[indx]));
	indx += (nx+2);

	event.wait();
	// auto submit_time = event.get_profiling_info<sycl::info::event_profiling::command_submit>();
	// auto start_time = event.get_profiling_info<sycl::info::event_profiling::command_start>();
	// auto end_time = event.get_profiling_info<sycl::info::event_profiling::command_end>();
	// submission_time += ((start_time - submit_time) / 1000000.0f);
	// execution_time += ((end_time - start_time) / 1000000.0f);
    }
    //std::cout << "Non-batched API (time in s) [kernel execution, submission] : " << execution_time << ", " << submission_time << std::endl;

    //copy input from device to host
    auto d2h = dev_que.memcpy(a, x_usm, n2ft3d*sizeof(double));
    d2h.wait();

    sycl::free(x_usm, dev_que);
}

// using batch version of oneMKL FFT
void fft_r2c_batch_onemkl(double* a) {
    sycl::queue dev_que(sycl::gpu_selector{},
			    sycl::property_list{sycl::property::queue::enable_profiling{},
				    sycl::property::queue::in_order{}});

    double* x_usm = sycl::malloc_device<double>(n2ft3d, dev_que);

    //copy input from host to device
    auto h2d = dev_que.memcpy(x_usm, a, n2ft3d*sizeof(double));
    h2d.wait();

    typedef oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
					 oneapi::mkl::dft::domain::REAL> descriptor_t;
    descriptor_t desc(nx);
    desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, nq1);
    desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, nx+2);
    desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, nx+2);
    desc.commit(dev_que);

    auto fft = compute_forward(desc, x_usm);

    fft.wait();
    // auto submit_time = fft.get_profiling_info<sycl::info::event_profiling::command_submit>();
    // auto start_time = fft.get_profiling_info<sycl::info::event_profiling::command_start>();
    // auto end_time = fft.get_profiling_info<sycl::info::event_profiling::command_end>();
    // auto submission_time = ((start_time - submit_time) / 1000000.0f);
    // auto execution_time = ((end_time - start_time) / 1000000.0f);
    // std::cout << "Batched API (time in s) [kernel execution, submission] : " << execution_time << ", " << submission_time << std::endl;

    //copy input from device to host
    auto d2h = dev_que.memcpy(a, x_usm, n2ft3d*sizeof(double));
    d2h.wait();

    sycl::free(x_usm, dev_que);
}

int main() {

    double* inout_fftpack = new double[n2ft3d];
    double* inout_nonbatch_onemkl = new double[n2ft3d];
    double* inout_batch_onemkl = new double[n2ft3d];

    const double itime = omp_get_wtime();

    init(inout_fftpack, n2ft3d/2);
    memcpy(inout_nonbatch_onemkl, inout_fftpack, n2ft3d*sizeof(double));
    memcpy(inout_batch_onemkl, inout_fftpack, n2ft3d*sizeof(double));

    fft_r2c_fftpack(inout_fftpack);
    fft_r2c_nonbatch_onemkl(inout_nonbatch_onemkl);
    fft_r2c_batch_onemkl(inout_batch_onemkl);

    for(int k=0; k<n2ft3d; k++) {
      if( std::abs(inout_fftpack[k] - inout_batch_onemkl[k]) > 1.0e-12 ) {
	std::cout << " output[" << k << "] : " << inout_fftpack[k] << ", " << inout_nonbatch_onemkl[k] <<  ", " << inout_batch_onemkl[k] << std::endl;
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
