#include <iostream>
#include <stdio.h>
#include <cmath>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <cstring>
#include <string>
#include <algorithm>
#include <random>
#include <numeric>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
//#include <thrust/sort.h>
//#include <thrust/execution_policy.h>

//bacteria surface geometry, UUU boundary semi-circle, Brownian noise
//Last updated: March 10, 2021

using namespace std;

//set random seed for uniform distribution for angles
unsigned int seed = time(NULL);
default_random_engine engine(seed);

#define THREADS_PER_BLOCK 128
#define TILE_SIZE 128
#define PI 3.14159265358979
#define K_B 1.38064852E-23 //m^2 kg s^-2 K^-1
#define DIM 3

//====================================================================================

//Returns the inverse parallel geometric factor for the
// translational friction tensor. a is the aspect ratio: a=l/d
__device__ double inverse_parallel_geo_factor(double a)
{
    double inverse_parallel = (log(a) - 0.207 + 0.980 / a - 0.133 / (a * a))
                                * (1.0 / (2.0 * PI * a));

    return inverse_parallel;
}

//Returns the inverse perpendicular geometric factor for the
//  translational friction tensor. a is the aspect ratio: a=l/d
__device__ double inverse_perpendicular_geo_factor(double a)
{
    double inverse_perp = (log(a) + 0.839 + 0.185 / a + 0.233 / (a * a))
                                * ( 1.0 / (4.0 * PI * a));

    return inverse_perp;
}

//Returns the rotational geometric factor for the
//  rotation friction tensor. a is the aspect ratio: a=l/d
__device__ double inverse_rotation_geo_factor(double a)
{
    double inverse_rotation = (log(a) - 0.662 + 0.917 / a - 0.050 / (a * a))
                                * ( 3.0 / (PI * a * a));

    return inverse_rotation;
}

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState *state) {

  int id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;

  /* Each thread gets same seed, a different sequence number, no offset */

  curand_init(seed, id, 0, &state[id]);
}

__global__ void init(unsigned int seed, curandStatePhilox4_32_10_t *state) {

  int id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;

  /* Each thread gets same seed, a different sequence number, no offset */

  curand_init(seed, id, 0, &state[id]);
}

__global__ void generate_random_numbers_noise(curandStatePhilox4_32_10_t *state, float4 *numbers) {

  int id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;

  /* Copy state to local memory for efficiency */
  curandStatePhilox4_32_10_t localState = state[id];

  numbers[id] = curand_normal4(&localState);

  /* Copy state back to global memory */
  state[id] = localState;
}

__device__ double ys_fcn(double4 xa, double xs, double R, double C)
{
  double ys;
  double param = (2.0 * R / PI) * acos(cos(PI * xs / (2.0 * R))) - R;
  if (xa.y > 0.0)
  {
    ys = sqrt(R * R - param * param) + C;
  }
  else
  {
    ys = -sqrt(R * R - param * param) - C;
  }

  return ys;
}

__device__ double dysdxs_fcn(double xs, double ys, double R)
{
  double dysdxs;

  double numer = R * (PI - 2.0 * acos(cos(PI * xs / (2.0 * R)))) * sin(PI * xs / (2.0 * R));
  double denom = sqrt(2.0) * sqrt(R * R * (PI - acos(cos(PI * xs / (2.0 * R)))) * acos(cos(PI * xs / (2.0 * R)))) *
  sqrt(1.0 - cos(PI * xs / R));

  if (ys > 0)
  {
    dysdxs =  numer / denom;
  }
  else
  {
    dysdxs =  -numer / denom;
  }
  return dysdxs;
}

__device__ double d2ysdxs2_fcn(double xs, double ys, double R)
{
  double d2ysdxs2;

  double numer = PI * PI * PI * R * R;

  double denom_param1 = (asin(cos(PI * xs / (2.0 * R)))) * (asin(cos(PI * xs / (2.0 * R))));
  double denom_param2 = sqrt(R * R * (PI * PI - 4.0 * denom_param1));
  double denom = denom_param2 * denom_param2 * denom_param2;

  if (ys > 0)
  {
    d2ysdxs2 = -numer / denom;
  }
  else
  {
    d2ysdxs2 = numer / denom;
  }
  return d2ysdxs2;
}

__device__ double func(double4 xa, double xs, double R, double C)
{
  double ys = ys_fcn(xa, xs, R, C);
  double dysdxs = dysdxs_fcn(xs, ys, R);

  double func_value = -xa.x + xs + (ys - xa.y) * dysdxs;
  return func_value;
}

__device__ double dfunc_dx(double4 xa, double xs, double R, double C)
{
  double ys = ys_fcn(xa, xs, R, C);
  double dysdxs = dysdxs_fcn(xs, ys, R);
  double d2ysdxs2 = d2ysdxs2_fcn(xs, ys, R);

  double dfunc_dx_value = 1.0 + dysdxs * dysdxs + (ys - xa.y) * d2ysdxs2;
  return dfunc_dx_value;
}

__device__ double3 SurfaceNormal(double xs, double ys, double R)
{
  double dysdxs = dysdxs_fcn( xs, ys, R );
  double denom = sqrt(dysdxs * dysdxs + 1.0);

  double dysplusdxs = dysdxs_fcn( xs, 1.0, R );

  double3 N;
  N.x = dysplusdxs / denom;
  N.z = 0.0;
  if (ys > 0.0)
  {
    N.y = -1.0 / denom;
  }
  else
  {
    N.y = 1.0 / denom;
  }

  return N;
}

__device__ double bisection(double xl, double xu, double4 xa, double R, double C)
{
  double es = 0.5; //percent
  int imax = 20;

  double fl = func(xa, xl, R, C);
  double fu = func(xa, xu, R, C);

  double xs;
  if (fl * fu < 0.0)
  {
    double ea = 1.0;
    int iter = 0;
    double xr = xl;
    double test, fr, xrold;
    while (iter < imax && ea > es)
    {
      xrold = xr;
      xr = (xl + xu) / 2.0;
      fr = func(xa, xr, R, C);

      test = fl * fr;
      ea = abs((xr - xrold) / xr) * 100.0;
      if (test < 0.0)
      {
        xu = xr;
      }
      else
      {
        xl = xr;
        fl = fr;
      }
      iter++;
    }
    xs = xr;
  }
  else
  {
    xs = sqrt(-1.0);
  }
  return xs;
}

__device__ bool isNaN(double s)
{
  // http.developer.nvidia.com/Cg/isnan.html
  return s != s;
}

__device__ double3 PointOnSurface(double4 xa, double R, double C)
{
  double xlower_bound, xupper_bound;
  double xl1, xu1, xs1, xl2, xu2, xs2, chck1, chck2, xs, ys;
  double3 S;

  //bounds are always set to be within the semi-circle the point is in
  double x_star = fmod(xa.x, (2.0 * R));
  xlower_bound = xa.x - 0.99 * x_star;
  xupper_bound = xa.x + 0.99 * ((2.0 * R) - x_star);

  xl1 = xlower_bound;
  xu1 = xa.x;

  xl2 = xa.x;
  xu2 = xupper_bound;

  //bisection on each section:
  xs1 = bisection(xl1, xu1, xa, R, C);
  xs2 = bisection(xl2, xu2, xa, R, C);

  //check roots
  if (isNaN(xs1) == 1)
  {
    chck1 = -1;
  }
  else
  {
    chck1 = dfunc_dx(xa, xs1, R, C);
  }

  if (isNaN(xs2) == 1)
  {
    chck2 = -1;
  }
  else
  {
    chck2 = dfunc_dx(xa, xs2, R, C);
  }

  if (chck1 > 0)
  {
    xs = xs1;
  }
  else if (chck2 > 0)
  {
    xs = xs2;
  }
  else
  {
    xs = xa.x;
  }

  ys = ys_fcn(xa, xs, R, C);

  S.x = xs;
  S.y = ys;
  S.z = 0.0;

  return S;
}

__device__ double2 dxsys_dxa(double3 S, double R, int i)
{
  double2 dxsysdxa;
  double dysdxs = dysdxs_fcn(S.x, S.y, R);

  if (i == 1)
  {
    dxsysdxa.x = 1.0;
    dxsysdxa.y = 1.0 / dysdxs;
  }
  else if (i == 2)
  {
    dxsysdxa.x = dysdxs;
    dxsysdxa.y = 1.0;
  }

  return dxsysdxa;
}

__device__ double2 dN_dxs(double xs, double ys, double R)
{
  double2 dNdxs;

  dNdxs.x = - 1.0 / R;

  double numer1 = sqrt(1.0 / (PI * PI - 4.0 * (asin(cos(PI * xs / (2.0 * R)))) * (asin(cos(PI * xs / (2.0 * R)))) ));
  double numer = 4.0 * asin(cos(PI * xs / (2.0 * R))) * numer1 * sin(PI * xs / (2.0 * R));
  double denom = R * sqrt(2.0 - 2.0 * cos(PI * xs / R));

  dNdxs.y = - numer / denom;

  if (ys < 0.0)
  {
    dNdxs.y = numer / denom;
  }

  return dNdxs;
}

__global__ void calculate_BodyWallInteraction(double3 *d_dUbdy_dxa,
  double3 *d_dUbdy_dna, double4 *d_x, double4 *d_n,
  double sigma_bdy, double R, double C, int N)
{
  	int gtid = blockIdx.x * blockDim.x + threadIdx.x;

  	if (gtid < N)
  	{
      double3 dUbdy_dxa, dUbdy_dna;

      dUbdy_dxa.x = 0.0;
      dUbdy_dxa.y = 0.0;
      dUbdy_dxa.z = 0.0;

      dUbdy_dna.x = 0.0;
      dUbdy_dna.y = 0.0;
      dUbdy_dna.z = 0.0;

      double4 xa = d_x[gtid];
  	  double4 na = d_n[gtid];

  	  double la = xa.w;
  	  double da = na.w;

      double y = ys_fcn(xa, xa.x, R, C);

      double chk1 = abs(xa.y);
      double chk2 = abs(y) - (la + da);

      if (chk1 > chk2)
      {
        double tol = 1.0 * da; //bacteria width
        double x_star = fmod(xa.x, (2.0 * R));

        if (x_star <= tol || ((2.0 * R) - x_star) <= tol) //bacteria is near peak
        {
          if (abs(1.0 - abs(na.y)) < 0.2) //bacteria near peak and vertical => treat like flat boundry
          {
            double3 S, W_hat;

            if (xa.y <= 0.0) //bottom surface
            {
              W_hat.x = 0.0;
              W_hat.y = 1.0;
              W_hat.z = 0.0;

              S.x = xa.x;
              S.y = -abs(C);
              S.z = 0.0;
            }
            else // top surface
            {
              W_hat.x = 0.0;
              W_hat.y = -1.0;
              W_hat.z = 0.0;

              S.x = xa.x;
              S.y = abs(C);
              S.z = 0.0;
            }

            double dot_na_W_hat, dot_xa_W_hat, dot_W_hat_S, r_alpha;
            dot_na_W_hat = na.x * W_hat.x + na.y * W_hat.y + na.z * W_hat.z;
            dot_xa_W_hat = xa.x * W_hat.x + xa.y * W_hat.y + xa.z * W_hat.z;
            dot_W_hat_S = W_hat.x * S.x + W_hat.y * S.y + W_hat.z * S.z;

            r_alpha = la * abs(dot_na_W_hat) + da - dot_xa_W_hat + dot_W_hat_S;

            double dUbdy_dralpha;
            double3 dralpha_dna;
            if (r_alpha > 0.0) //contact with boundary
            {
              dUbdy_dralpha = 0.01 * (1.0 / sigma_bdy) * exp(r_alpha / sigma_bdy);
              //0.01 factor to reduce the effect of the flat boundary

              //boundary force derivatives:
              dUbdy_dxa.x = dUbdy_dralpha * -W_hat.x;
              dUbdy_dxa.y = dUbdy_dralpha * -W_hat.y;
              dUbdy_dxa.z = dUbdy_dralpha * -W_hat.z;

              //boundary orientation derivatives:
              if (dot_na_W_hat == 0.0)
              {
                dUbdy_dna.x = 0.0;
                dUbdy_dna.y = 0.0;
                dUbdy_dna.z = 0.0;
              }
              else
              {
                dralpha_dna.x = (la * dot_na_W_hat / abs(dot_na_W_hat)) * W_hat.x;
                dralpha_dna.y = (la * dot_na_W_hat / abs(dot_na_W_hat)) * W_hat.y;
                dralpha_dna.z = (la * dot_na_W_hat / abs(dot_na_W_hat)) * W_hat.z;

                dUbdy_dna.x = dUbdy_dralpha * dralpha_dna.x;
                dUbdy_dna.y = dUbdy_dralpha * dralpha_dna.y;
                dUbdy_dna.z = dUbdy_dralpha * dralpha_dna.z;
              }
            }
          }
        }
        else
        {
          double3 S, Nhat; // point on the surface closest to bacteria
          S = PointOnSurface(xa, R, C);
          Nhat = SurfaceNormal(S.x, S.y, R);

          double3 xa_S;
          xa_S.x = xa.x - S.x;
          xa_S.y = xa.y - S.y;
          xa_S.z = xa.z - S.z;

          double dot_na_Nhat, dot_Nhat_xa_S, r_alpha;

          dot_na_Nhat = na.x * Nhat.x + na.y * Nhat.y + na.z * Nhat.z;
          dot_Nhat_xa_S = Nhat.x * xa_S.x + Nhat.y * xa_S.y + Nhat.z * xa_S.z;

          r_alpha = la * abs(dot_na_Nhat) + da - dot_Nhat_xa_S;

    	    if (r_alpha > 0.0) //contact with boundary
    	    {
            double dUbdy_dralpha;
            double3 dralpha_dna, dralpha_dxa;

            dUbdy_dralpha = (1.0 / sigma_bdy) * exp(r_alpha / sigma_bdy);

            double2 dNdxs = dN_dxs(S.x, S.y, R);
            double2 dxys_dxa1 = dxsys_dxa(S, R, 1);
            double2 dxys_dxa2 = dxsys_dxa(S, R, 2);
            double dysdxs = dysdxs_fcn(S.x, S.y, R);
            double dxsdys = 1.0 / dysdxs;

            double c1 = la * dot_na_Nhat / abs(dot_na_Nhat);
            double c2 = (xa.x - S.x) * dNdxs.x;
            double c3 = (xa.y - S.y) * dNdxs.y;

            dralpha_dxa.x = c1 * (na.x * dNdxs.x + na.y * dNdxs.y) * dxys_dxa1.x
              - (c2 + c3) * dxys_dxa1.x
              - Nhat.x
              + (Nhat.x + Nhat.y * dysdxs) * dxys_dxa1.x
              + (Nhat.x * dxsdys + Nhat.y) * dxys_dxa1.y;

            dralpha_dxa.y = c1 * (na.x * dNdxs.x + na.y * dNdxs.y) * dxys_dxa2.x
              - (c2 + c3) * dxys_dxa2.x
              - Nhat.y
              + (Nhat.x + Nhat.y * dysdxs) * dxys_dxa2.x
              + (Nhat.x * dxsdys + Nhat.y) * dxys_dxa2.y;

              dralpha_dxa.z = 0.0;

            //boundary force derivatives:
            dUbdy_dxa.x = dUbdy_dralpha * dralpha_dxa.x;
            dUbdy_dxa.y = dUbdy_dralpha * dralpha_dxa.y;
            dUbdy_dxa.z = dUbdy_dralpha * dralpha_dxa.z;

            //boundary orientation derivatives:
            if (dot_na_Nhat == 0.0)
            {
              dUbdy_dna.x = 0.0;
              dUbdy_dna.y = 0.0;
              dUbdy_dna.z = 0.0;
            }
            else
            {
              dralpha_dna.x = c1 * Nhat.x;
              dralpha_dna.y = c1 * Nhat.y;
              dralpha_dna.z = c1 * Nhat.z;

              dUbdy_dna.x = dUbdy_dralpha * dralpha_dna.x;
              dUbdy_dna.y = dUbdy_dralpha * dralpha_dna.y;
              dUbdy_dna.z = dUbdy_dralpha * dralpha_dna.z;
            }
    	    }
        }
      }

  		// Save the result in global memory for the integration step
  		d_dUbdy_dxa[gtid] = dUbdy_dxa;
  		d_dUbdy_dna[gtid] = dUbdy_dna;
    }
}

__global__ void time_marching(double4 *d_x, double4 *d_n,
  double3 *d_dUbdy_dxa, double3 *d_dUbdy_dna,
  double epsilon_r,
  double inverse_Pe_T, double inverse_Pe_parallel, double inverse_Pe_perp, double inverse_Pe_R,
  double dt, int N, double L,
  double *d_t_run, double *d_t_tumble, int *d_tumble_flag,
  double delta_run, double delta_tumble,
  double avg_n_tumble, double std_n_tumble,
  curandState *state, float4 *d_random_numbers_noise)
{
	int gtid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gtid < N)
	{
		double4 xa = d_x[gtid];
		double4 na = d_n[gtid];
    float4 random_numbers_noise = d_random_numbers_noise[gtid];

		double la = xa.w;
		double da = na.w;

		double3 dUbdy_dxa = d_dUbdy_dxa[gtid];
		double3 dUbdy_dna = d_dUbdy_dna[gtid];

    double4 x_new;
    double4 n_new;

    //-----Start: creating orientation and orientation projection matrix-----
    double ori_matrix[DIM][DIM];
    ori_matrix[0][0] = na.x * na.x;
    ori_matrix[1][1] = na.y * na.y;
    ori_matrix[2][2] = na.z * na.z;
    ori_matrix[0][1] = na.x * na.y;
    ori_matrix[0][2] = na.x * na.z;
    ori_matrix[1][2] = na.y * na.z;
    ori_matrix[1][0] = ori_matrix[0][1];
    ori_matrix[2][0] = ori_matrix[0][2];
    ori_matrix[2][1] = ori_matrix[1][2];

    double ori_proj_matrix[DIM][DIM];
    ori_proj_matrix[0][0] = 1.0 - na.x * na.x;
    ori_proj_matrix[1][1] = 1.0 - na.y * na.y;
    ori_proj_matrix[2][2] = 1.0 - na.z * na.z;
    ori_proj_matrix[0][1] = 0.0 - na.x * na.y;
    ori_proj_matrix[0][2] = 0.0 - na.x * na.z;
    ori_proj_matrix[1][2] = 0.0 - na.y * na.z;
    ori_proj_matrix[1][0] = ori_proj_matrix[0][1];
    ori_proj_matrix[2][0] = ori_proj_matrix[0][2];
    ori_proj_matrix[2][1] = ori_proj_matrix[1][2];
    //-----End: creating orientation and orientation projection matrix-----

    //-----Start: time-marching + tumbling dynamics-----
    if (d_tumble_flag[gtid] == 1) //tumbling
    {
      d_t_tumble[gtid] += dt;

      if (d_t_tumble[gtid] < delta_tumble) //don't move
      {
        x_new.x = xa.x;
        x_new.y = xa.y;
        x_new.z = xa.z;
        x_new.w = xa.w;

        n_new.x = na.x;
        n_new.y = na.y;
        n_new.z = na.z;
        n_new.w = na.w;
      }
      else //tumble
      {
        d_tumble_flag[gtid] = 0;
        d_t_tumble[gtid] = 0.0;

        float angle;
        double rad_angle;

        curandState localState = state[gtid];

        angle = curand_normal(&localState);
        angle = angle * std_n_tumble + avg_n_tumble;
        while (angle < 0.0 || angle > 180.0)
        {
          angle = curand_normal(&localState);
          angle = angle * std_n_tumble + avg_n_tumble;
        }
        double uniform1 = curand_uniform(&localState); //number between 0 and 1
        if (uniform1 < 0.5) //otherwise angle is positive
        {
          angle = -angle;
        }

        state[gtid] = localState;

        rad_angle = angle * PI / 180; //convert to radians

        //rotation matrix
        double R[2][2];
        R[0][0] = cos(rad_angle);
        R[0][1] = -sin(rad_angle);
        R[1][0] = sin(rad_angle);
        R[1][1] = cos(rad_angle);

        n_new.x = R[0][0] * na.x + R[0][1] * na.y;
        n_new.y = R[1][0] * na.x + R[1][1] * na.y;
        n_new.z = 0.0;
        n_new.w = da;

        x_new.x = xa.x;
        x_new.y = xa.y;
        x_new.z = xa.z;
        x_new.w = xa.w;
      }
    }
    else //run
    {
      d_t_run[gtid] += dt;

      if (d_t_run[gtid] < delta_run) //run
      {
        //translational dynamics:

        //calculating geometric factors:
    		double aspect = la/da;
    		double inverse_parallel = inverse_parallel_geo_factor(aspect);
    		double inverse_perp = inverse_perpendicular_geo_factor(aspect);
        double inverse_rotation = inverse_rotation_geo_factor(aspect);

    		//-----Start: creating Gamma_inverse matrix-----
    		double Gamma_inverse[DIM][DIM];
    		for(int i = 0; i < DIM; i++)
    		{
    			for(int j = 0; j < DIM; j++)
    			{
    				Gamma_inverse[i][j] = inverse_parallel * ori_matrix[i][j]
                                  + inverse_perp * ori_proj_matrix[i][j];
    			}
    		}
    		//-----End: creating Gamma_inverse matrix-----

        //-----Start: creating translational diffusion matrix-----
        double Pe_trans_matrix[DIM][DIM];
        double sqrt_Pe_inverse_parallel = sqrt(inverse_Pe_parallel);
        double sqrt_Pe_inverse_perp = sqrt(inverse_Pe_perp);

        for(int i = 0; i < DIM; i++)
        {
          for(int j = 0; j < DIM; j++)
          {
            Pe_trans_matrix[i][j] = sqrt_Pe_inverse_parallel * ori_matrix[i][j]
                                  + sqrt_Pe_inverse_perp * ori_proj_matrix[i][j];
          }
        }
        //-----End: creating translational diffusion matrix-----

    		//adding it all together:
    		double3 x_b;
    		x_b.x = - epsilon_r * (inverse_Pe_T) * dUbdy_dxa.x;
    		x_b.y = - epsilon_r * (inverse_Pe_T) * dUbdy_dxa.y;
    		x_b.z = - epsilon_r * (inverse_Pe_T) * dUbdy_dxa.z;

    		//matrix multiply:
    		double3 Gamma_inverse_x_b;
    		Gamma_inverse_x_b.x = Gamma_inverse[0][0] * x_b.x
                            + Gamma_inverse[0][1] * x_b.y
                            + Gamma_inverse[0][2] * x_b.z;
    		Gamma_inverse_x_b.y = Gamma_inverse[1][0] * x_b.x
                            + Gamma_inverse[1][1] * x_b.y
                            + Gamma_inverse[1][2] * x_b.z;
    		Gamma_inverse_x_b.z = Gamma_inverse[2][0] * x_b.x
                            + Gamma_inverse[2][1] * x_b.y
                            + Gamma_inverse[2][2] * x_b.z;

        //noise:
        float3 d_xi;
        d_xi.x = random_numbers_noise.x * sqrt(2.0 * dt);
        d_xi.y = random_numbers_noise.y * sqrt(2.0 * dt);
        d_xi.z = 0.0;

        float3 trans_noise;
        trans_noise.x = Pe_trans_matrix[0][0] * d_xi.x
                      + Pe_trans_matrix[0][1] * d_xi.y
                      + Pe_trans_matrix[0][2] * d_xi.z;
        trans_noise.y = Pe_trans_matrix[1][0] * d_xi.x
                      + Pe_trans_matrix[1][1] * d_xi.y
                      + Pe_trans_matrix[1][2] * d_xi.z;
        trans_noise.z = Pe_trans_matrix[2][0] * d_xi.x
                      + Pe_trans_matrix[2][1] * d_xi.y
                      + Pe_trans_matrix[2][2] * d_xi.z;


        //time step:
    		x_new.x = xa.x + na.x * dt + Gamma_inverse_x_b.x * dt + trans_noise.x;
    		x_new.y = xa.y + na.y * dt + Gamma_inverse_x_b.y * dt + trans_noise.y;
    		x_new.z = 0.0;
    		x_new.w = la;

        //orientation dynamics
        double3 n_b;
        int dim = 2;

    		n_b.x = - epsilon_r * (inverse_Pe_R) * inverse_rotation * dUbdy_dna.x + (1 - dim) * (inverse_Pe_R) * na.x;
    		n_b.y = - epsilon_r * (inverse_Pe_R) * inverse_rotation * dUbdy_dna.y + (1 - dim) * (inverse_Pe_R) * na.y;
    		n_b.z = - epsilon_r * (inverse_Pe_R) * inverse_rotation * dUbdy_dna.z + (1 - dim) * (inverse_Pe_R) * na.z;

        double3 ori_proj_n_b;
        ori_proj_n_b.x = ori_proj_matrix[0][0] * n_b.x
                       + ori_proj_matrix[0][1] * n_b.y
                       + ori_proj_matrix[0][2] * n_b.z;
        ori_proj_n_b.y = ori_proj_matrix[1][0] * n_b.x
                       + ori_proj_matrix[1][1] * n_b.y
                       + ori_proj_matrix[1][2] * n_b.z;
        ori_proj_n_b.z = ori_proj_matrix[2][0] * n_b.x
                       + ori_proj_matrix[2][1] * n_b.y
                       + ori_proj_matrix[2][2] * n_b.z;

        //noise:
        float3 d_zeta;
        d_zeta.x = random_numbers_noise.z * sqrt(2.0 * (inverse_Pe_R) * dt);
        d_zeta.y = random_numbers_noise.w * sqrt(2.0 * (inverse_Pe_R) * dt);
        d_zeta.z = 0.0;

        double3 ori_noise;
        ori_noise.x = ori_proj_matrix[0][0] * d_zeta.x
                    + ori_proj_matrix[0][1] * d_zeta.y
                    + ori_proj_matrix[0][2] * d_zeta.z;
        ori_noise.y = ori_proj_matrix[1][0] * d_zeta.x
                    + ori_proj_matrix[1][1] * d_zeta.y
                    + ori_proj_matrix[1][2] * d_zeta.z;
        ori_noise.z = ori_proj_matrix[2][0] * d_zeta.x
                    + ori_proj_matrix[2][1] * d_zeta.y
                    + ori_proj_matrix[2][2] * d_zeta.z;

        n_new.x = na.x + ori_proj_n_b.x * dt + ori_noise.x;
        n_new.y = na.y + ori_proj_n_b.y * dt + ori_noise.y;
        n_new.z = 0.0;
        n_new.w = da;
      }
      else
      {
        d_tumble_flag[gtid] = 1;
        d_t_run[gtid] = 0.0;

        x_new.x = xa.x;
        x_new.y = xa.y;
        x_new.z = xa.z;
        x_new.w = xa.w;

        n_new.x = na.x;
        n_new.y = na.y;
        n_new.z = na.z;
        n_new.w = na.w;
      }
    }
    //-----End: time-marching + tumbling dynamics-----

    //normalize n afterwards:
    double magn_n_new_Sqrd = n_new.x * n_new.x + n_new.y * n_new.y + n_new.z * n_new.z;
    double magn_n_new = sqrt(magn_n_new_Sqrd);

    n_new.x = (n_new.x / magn_n_new);
    n_new.y = (n_new.y / magn_n_new);
    n_new.z = (n_new.z / magn_n_new);

    //periodic BC
    if (x_new.x < 0.0)
    {
      x_new.x = L - x_new.x;
    }
    else if (x_new.x > L)
    {
      double delta_x = x_new.x - L;
      x_new.x = delta_x;
    }

		// Save the result in global memory
		d_x[gtid] = x_new;
		d_n[gtid] = n_new;
	}
}

//returns the greater common divisor of two numbers
int gcd(int first_number, int second_number)
{
	int gcd_value;

	for(int i = 1; i <= first_number && i <= second_number; i++)
	{
		if(first_number % i == 0 && second_number % i == 0 )
		{
			gcd_value = i;
		}

	}
	return gcd_value;
}

//loads the .txt file that contains the simulation input variables data
void load_textfile_sim_parameters( char filename[],
  int& sim_num, int& case_num,
  double& dt, double& time_save, double& start_time, double& final_time,
  int& N, double& l, double& d,
  double& C, double& L, double& R,
  double& epsilon_r, double& sigma_bdy,
  double& inverse_Pe_T, double& inverse_Pe_parallel, double& inverse_Pe_perp, double& inverse_Pe_R,
  double& delta_run, double& delta_tumble, double& avg_n_tumble, double& std_n_tumble)
{
	ifstream infile(filename);
	if (infile.fail())
    {
        cout<<"\nSimulation parameters input file opening failed.\n";
        exit(1);
    }

  int number_inputs = 22;
  double input_vec[number_inputs];

  for (int i = 0; i < number_inputs; i++)
  {
    infile >> input_vec[i];
  }

  int i = 0;
  sim_num = int(input_vec[i]);
  case_num = int(input_vec[++i]);
  dt = input_vec[++i];
  time_save = input_vec[++i];
  start_time = input_vec[++i];
  final_time = input_vec[++i];
  N = int(input_vec[++i]);
  l = input_vec[++i];
  d = input_vec[++i];
  C = input_vec[++i];
  L = input_vec[++i];
  R = input_vec[++i];
  epsilon_r = input_vec[++i];
  sigma_bdy = input_vec[++i];
  inverse_Pe_T = input_vec[++i];
  inverse_Pe_parallel = input_vec[++i];
  inverse_Pe_perp = input_vec[++i];
  inverse_Pe_R = input_vec[++i];
  delta_run = input_vec[++i];
  delta_tumble = input_vec[++i];
  avg_n_tumble = input_vec[++i];
  std_n_tumble = input_vec[++i];

	cout << "\nSimulation parameters loaded\n";
}

void initial_loading(double4 x[], double4 n[], int N, double C, double L,
  double l, double d, double t_run[], double t_tumble[], double delta_run, double simulation_time)
{
  double factorL = 1.0;

  double factorLminus1 = 1.0 - factorL;

  double xmin = 0.0 + 0.5 * factorLminus1 * L;
  double xmax = L - 0.5 * factorLminus1 * L;

  double ymin = -C;
  double ymax = C;

  uniform_real_distribution<double> uniform_x(xmin, xmax);
  uniform_real_distribution<double> uniform_y(ymin, ymax);
  uniform_real_distribution<double> uniform_dist_angle(0, 2.0 * PI );
  uniform_real_distribution<double> uniform_dist_run_time(0.0, delta_run);

  double angle;
  for(int alpha = 0; alpha < N; alpha++)
  {
    //set bacteria dimensions:
    x[alpha].w = l;
    n[alpha].w = d;

    //set initial positions
    x[alpha].x = uniform_x(engine);
    x[alpha].y = uniform_y(engine);
    x[alpha].z = 0.0;

    //set initial bacteria orientations:
    angle = uniform_dist_angle(engine);
    n[alpha].x = cos(angle);
    n[alpha].y = sin(angle);
    n[alpha].z = 0.0;

    //set initial run time
    if (delta_run < simulation_time) {
      t_run[alpha] = uniform_dist_run_time(engine);
    }
    else {
      t_run[alpha] = 0.0;
    }

    //set initial tumble time
    t_tumble[alpha] = 0.0;
  }
  return;
}

//Returns the eigenvectors corresponding to the orientation vectors for
//  all the bacteria.
void eigenvectors_ellipsoid(double eigenvectors[][DIM*DIM], double4 n[], int N)
{
    for (int alpha = 0; alpha < N; alpha++)
    {
        if (n[alpha].x == 1.0)
        {
            //v1:
            eigenvectors[alpha][0] = 1.0;
            eigenvectors[alpha][1] = 0.0;
            eigenvectors[alpha][2] = 0.0;

            //v2:
            eigenvectors[alpha][3] = 0.0;
            eigenvectors[alpha][4] = 1.0;
            eigenvectors[alpha][5] = 0.0;

            //v3:
            eigenvectors[alpha][6] = 0.0;
            eigenvectors[alpha][7] = 0.0;
            eigenvectors[alpha][8] = 1.0;
        }
        else if (n[alpha].x == -1.0)
        {
            //v1:
            eigenvectors[alpha][0] = -1.0;
            eigenvectors[alpha][1] = 0.0;
            eigenvectors[alpha][2] = 0.0;

            //v2:
            eigenvectors[alpha][3] = 0.0;
            eigenvectors[alpha][4] = -1.0;
            eigenvectors[alpha][5] = 0.0;

            //v3:
            eigenvectors[alpha][6] = 0.0;
            eigenvectors[alpha][7] = 0.0;
            eigenvectors[alpha][8] = 1.0;
        }
        else
        {
            //v1:
            eigenvectors[alpha][0] = n[alpha].x;
            eigenvectors[alpha][1] = n[alpha].y;
            eigenvectors[alpha][2] = n[alpha].z;

            double denom = sqrt(1.0 - n[alpha].x * n[alpha].x );
            //v2:
            eigenvectors[alpha][3] = 0.0;
            eigenvectors[alpha][4] = -n[alpha].z / denom;
            eigenvectors[alpha][5] = n[alpha].y / denom;

            //v3:
            eigenvectors[alpha][6] = 1.0 - n[alpha].x * n[alpha].x;
            eigenvectors[alpha][7] = -(n[alpha].x * n[alpha].y) / denom;
            eigenvectors[alpha][8] = -(n[alpha].x * n[alpha].z) / denom;
        }
    }
    return;
}

//Prints simulation input to file
void print_to_file_input(
  int sim_num, int case_num,
  double dt, double time_save, double start_time, double final_time,
  int N, double l, double d,
  double C, double L, double R,
  double epsilon_r, double sigma_bdy,
  double inverse_Pe_T, double inverse_Pe_parallel, double inverse_Pe_perp, double inverse_Pe_R,
  double delta_run, double delta_tumble, double avg_n_tumble, double std_n_tumble)
{
  ofstream fout;
  char file_name2[100];
  sprintf(file_name2,"SimulationInput.txt");
  fout.open(file_name2);
  if (fout.fail())
  {
    cout<<"Output file opening failed.\n";
    exit(1);
  }

  fout.setf(ios::fixed);
  fout.setf(ios::showpoint);
  fout.precision(30);

  string headers("sim_num, case_num, dt, time_save, start_time, final_time, N, l, d, C, L, R, epsilon_r, sigma_bdy, inverse_Pe_T, inverse_Pe_parallel, inverse_Pe_perp, inverse_Pe_R, delta_run, delta_tumble, avg_n_tumble, std_n_tumble");

  fout << headers << endl;
  fout << sim_num << ", "
       << case_num << ", "
       << dt << ", "
       << time_save << ", "
       << start_time << ", "
       << final_time << ", "
       << N << ", "
       << l << ", "
       << d << ", "
       << C << ", "
       << L << ", "
       << R << ", "
       << epsilon_r << ", "
       << sigma_bdy << ", "
       << inverse_Pe_T << ", "
       << inverse_Pe_parallel << ", "
       << inverse_Pe_perp << ", "
       << inverse_Pe_R << ", "
       << delta_run << ", "
       << delta_tumble << ", "
       << avg_n_tumble << ", "
       << std_n_tumble << endl;

       fout.close();
       return;
}

//Prints output to file
void print_to_file_output(int sim_num, int case_num, int itime, int N,
  double4 x[], double4 n[], double t_run[])
{
		double eig_vec[N][DIM * DIM];    //dimensionless Cartesian vector components of the eigenvectors for the orientation of the bacteria
		eigenvectors_ellipsoid(eig_vec, n, N);

		ofstream fout;
		char file_name2[100];
		sprintf(file_name2,"sim%d_case%d_timestep%015d.txt", sim_num, case_num, itime);
		fout.open(file_name2);
		if (fout.fail())
		{
			cout<<"Output file opening failed.\n";
			exit(1);
		}

		fout.setf(ios::fixed);
    fout.setf(ios::showpoint);
    fout.precision(15);

    string headers("Centroid_1, Centroid_2, Centroid_3, DirVector1_1, DirVector1_2, DirVector1_3, DirVector2_1, DirVector2_2, DirVector2_3, DirVector3_1, DirVector3_2, DirVector3_3, SemiAxis1, SemiAxis2, SemiAxis3, tRun");

    fout << headers << endl;
    for (int alpha = 0; alpha < N; alpha++)
		{
			  fout << x[alpha].x << ", "
				     << x[alpha].y << ", "
				     << x[alpha].z << ", ";

			  for (int nCol = 0; nCol < DIM*DIM; nCol++)
        {
             fout << eig_vec[alpha][nCol] << ", ";
        }

			  fout << x[alpha].w << ", "
				     << n[alpha].w << ", "
				     << n[alpha].w << ", "
             << t_run[alpha] << endl;
		   }

		fout.close();

    return;
}

//====================================================================================

int main(void)
{
	  //-----Start: simulation input-----
    int sim_num;                     //simulation number
    int case_num;                     //case number

    double dt;                //dimensionless time step
    double time_save;         //dimensionless time at which to output
    double start_time;		 //dimensionless start time of simulation
    double final_time;		 //dimensionless final time of simulation

    int N;                        //number of bacteria in simulation

    double l;       //half-length of bacteria
    double d;       //half-diameter of bacteria

    double C;       //wall surface displacement from origin
    double L;       //wall length (a multiple of lambda)
    double R;

    double epsilon_r;
	  double sigma_bdy;				 //range parameter for bacteria-wall steric repulsion

    double inverse_Pe_T;
    double inverse_Pe_parallel;
    double inverse_Pe_perp;
    double inverse_Pe_R;

	  double delta_run;					//run time
    double delta_tumble;      //tumble time
	  double avg_n_tumble;		//average tumbling angle in degrees
	  double std_n_tumble;		//std tumbling angle in degrees

    load_textfile_sim_parameters( "bacteria_surface_input.txt",
    sim_num, case_num,
    dt, time_save, start_time, final_time,
    N, l, d,
    C, L, R,
    epsilon_r, sigma_bdy,
    inverse_Pe_T, inverse_Pe_parallel, inverse_Pe_perp, inverse_Pe_R,
    delta_run, delta_tumble, avg_n_tumble, std_n_tumble);

    L = L * 2.0 * R;

    cout.setf(ios::fixed);
    cout.setf(ios::showpoint);
    cout.precision(15);
    cout << endl<<"==============="<<endl
         << "sim_num = " << sim_num << endl
         << "case_num = " << case_num << endl
         << "dt = " << dt << endl
         << "time_save = " << time_save << endl
         << "start_time = " << start_time << endl
         << "final_time = " << final_time << endl
         << "N = " << N << endl
         << "l = " << l << endl
         << "d = " << d << endl
         << "C = " << C << endl
         << "L = " << L << endl
         << "R = " << R << endl
         << "epsilon_r = " << epsilon_r << endl
         << "sigma_bdy = " << sigma_bdy << endl
         << "inverse_Pe_T = " << inverse_Pe_T << endl
         << "inverse_Pe_parallel = " << inverse_Pe_parallel << endl
         << "inverse_Pe_perp = " << inverse_Pe_perp << endl
         << "inverse_Pe_R = " << inverse_Pe_R << endl
         << "delta_run = "<< delta_run << endl
         << "delta_tumble = " << delta_tumble << endl
         << "avg_n_tumble = " << avg_n_tumble << endl
         << "std_n_tumble = " << std_n_tumble << endl
         << "================"<<endl;
    cout.precision(15);

    print_to_file_input(sim_num, case_num, dt, time_save, start_time, final_time,
      N, l, d, C, L, R, epsilon_r, sigma_bdy,
      inverse_Pe_T, inverse_Pe_parallel, inverse_Pe_perp, inverse_Pe_R, delta_run, delta_tumble, avg_n_tumble, std_n_tumble);
  //-----End: simulation input-----

	  //-----Start: declaring derived simulation parameters-----
    //simulation variables:
		int time_steps = ceil((final_time - start_time) / dt);              //number of simulation time steps
		int timestep_save; //number of simulation time steps until save output
    timestep_save = ceil(time_save / dt);

    double4 x[N];                    //dimensionless Cartesian coordinates of the bacteria & dimensionless half-length of the bacteria
    double4 n[N];                    //dimensionless Cartesian vector components of the orientation vector of the bacteria & dimensionless half-diameter of the bacteria
    double t_run[N];              //run time of the bacteria
    double t_tumble[N];           //tumble time of bacteria
    int tumble_flag[N];           //tumble flag of bacteria (if tumble_flag[alpha] = 1, then bacteria tumbles; otherwise it runs)

    memset(x, 0, N * sizeof(double4));
		memset(n, 0, N * sizeof(double4));
    memset(t_run, 0, N * sizeof(double));
    memset(t_tumble, 0, N * sizeof(double));
    memset(tumble_flag, 0, N * sizeof(int));
    //-----End: declaring derived simulation parameters-----

    //-----Start: INITIALIZING-----

    //-----Start: initial positions, orientations, and run time-----
    initial_loading(x, n, N, C, L, l, d, t_run, t_tumble, delta_run, (final_time - start_time));
    //-----End: initial positions, orientations, and run time-----

    //-----Start: print initial positions and orientations-----
    print_to_file_output(sim_num, case_num, 0, N, x, n, t_run);
    //-----End: print initial positions and orientations-----

		//-----Start: set up cuda variables-----
    // calculate number of blocks and threads needed
    int num_BLOCKS, num_THREADS;
    if (N < THREADS_PER_BLOCK)
    {
      num_BLOCKS = 1;
      num_THREADS = N;
    }
    else
    {
      num_BLOCKS = 1 + (N - 1)/THREADS_PER_BLOCK; //ceiling, use only if h_N != 0
      num_THREADS = THREADS_PER_BLOCK;
    }

		// declare GPU memory pointers
		double4 *d_x;
		double4 *d_n;
		double3 *d_dUbdy_dxa;
		double3 *d_dUbdy_dna;
		double *d_t_run;
    double *d_t_tumble;
    int *d_tumble_flag;
    float4 *d_random_numbers_noise;

		// allocate GPU memory
		cudaMalloc((void**) &d_x, N * sizeof(double4));
		cudaMalloc((void**) &d_n, N * sizeof(double4));
		cudaMalloc((void**) &d_dUbdy_dxa, N * sizeof(double3));
		cudaMalloc((void**) &d_dUbdy_dna, N * sizeof(double3));
		cudaMalloc((void**) &d_t_run, N * sizeof(double));
    cudaMalloc((void**) &d_t_tumble, N * sizeof(double));
    cudaMalloc((void**) &d_tumble_flag, N * sizeof(int));
    cudaMalloc((void **)&d_random_numbers_noise, N * sizeof(float4));

		// transfer the array to the GPU
		cudaMemcpy(d_x, x, N * sizeof(double4), cudaMemcpyHostToDevice);
		cudaMemcpy(d_n, n, N * sizeof(double4), cudaMemcpyHostToDevice);
		cudaMemcpy(d_t_run, t_run, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t_tumble, t_tumble, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tumble_flag, tumble_flag, N * sizeof(int), cudaMemcpyHostToDevice);

    //random number generators:
    curandState *d_CurandStates;
    curandStatePhilox4_32_10_t *d_PHILOXStates;
    cudaMalloc((void **) &d_CurandStates, N * sizeof(curandState));
		cudaMalloc((void **) &d_PHILOXStates, N * sizeof(curandStatePhilox4_32_10_t));
		// setup seeds
    init<<< num_BLOCKS, num_THREADS >>>(seed, d_CurandStates);
    init<<< num_BLOCKS, num_THREADS >>>(seed, d_PHILOXStates);
		//-----End: set up cuda variables-----
    cout << "End: INITIALIZING" << endl;
    //-----End: INITIALIZING-----

		//-----Start: DYNAMICS LOOP-----
		int itime = floor(start_time / dt) + 1;

    cout << "itime: " << itime << endl;
    cout << "time_steps: " << time_steps << endl;

		while (itime <= time_steps)
		{
      //-----Start: random numbers -----
      generate_random_numbers_noise<<< num_BLOCKS, num_THREADS >>>(d_PHILOXStates, d_random_numbers_noise);
      //-----End: random numbers -----

      //-----Start: boundary interactions-----
      calculate_BodyWallInteraction<<< num_BLOCKS, num_THREADS >>>(d_dUbdy_dxa,
        d_dUbdy_dna, d_x, d_n,
        sigma_bdy, R, C, N);
			//-----End: boundary interactions-----

			//-----Start: time-marching-----
      time_marching<<< num_BLOCKS, num_THREADS >>>
      (d_x, d_n,
        d_dUbdy_dxa, d_dUbdy_dna,
        epsilon_r,
        inverse_Pe_T, inverse_Pe_parallel, inverse_Pe_perp, inverse_Pe_R,
        dt, N, L,
        d_t_run, d_t_tumble, d_tumble_flag,
        delta_run, delta_tumble,
        avg_n_tumble, std_n_tumble,
        d_CurandStates, d_random_numbers_noise);
			//-----End: time-marching-----

			//-----Start: saving variables-----
			if ( itime % timestep_save == 0)
			{
			 // copy back the result array to the CPU
			 cudaMemcpy(x, d_x, N * sizeof(double4), cudaMemcpyDeviceToHost);
			 cudaMemcpy(n, d_n, N * sizeof(double4), cudaMemcpyDeviceToHost);
       cudaMemcpy(t_run, d_t_run, N * sizeof(double), cudaMemcpyDeviceToHost);

			 print_to_file_output(sim_num, case_num, itime, N, x, n, t_run);
			}
			//-----End: saving variables-----

			printf("\ntime step: %d", itime);
			itime++;
		}
		cout << endl << endl;
		//-----End: DYNAMICS LOOP-----

	return 0;
}
