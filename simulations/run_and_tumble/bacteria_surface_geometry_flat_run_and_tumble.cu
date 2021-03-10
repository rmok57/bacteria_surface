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

//bacteria surface geometry, flat surface, run and tumble
//last updated: March 9, 2021

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

__global__ void calculate_BodyWallInteraction(double3 *d_dUbdy_dxa,
  double3 *d_dUbdy_dna, double4 *d_x, double4 *d_n, double3 W_hat,
  double C, double sigma_bdy, int N)
{
  	int gtid = blockIdx.x * blockDim.x + threadIdx.x;

  	if (gtid < N)
  	{
  		double4 xa = d_x[gtid];
  		double4 na = d_n[gtid];

  		double la = xa.w;
  		double da = na.w;
      double3 S; // point on the surface closest to bacteria

      if (xa.y <= 0.0) //bottom surface
      {
        W_hat.x = abs(W_hat.x);
        W_hat.y = abs(W_hat.y);
        W_hat.z = abs(W_hat.z);

        S.x = xa.x;
        S.y = -abs(C);
        S.z = 0.0;
      }
      else // top surface
      {
        W_hat.x = -abs(W_hat.x);
        W_hat.y = -abs(W_hat.y);
        W_hat.z = -abs(W_hat.z);

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
      double3 dUbdy_dxa, dUbdy_dna;
      double3 dralpha_dna;
  		if (r_alpha > 0.0) //contact with boundary
  		{
  			dUbdy_dralpha = (1.0 / sigma_bdy) * exp(r_alpha / sigma_bdy);

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
  		else //no contact with boundary
  		{
  			dUbdy_dxa.x = 0.0;
        dUbdy_dxa.y = 0.0;
        dUbdy_dxa.z = 0.0;

        dUbdy_dna.x = 0.0;
        dUbdy_dna.y = 0.0;
        dUbdy_dna.z = 0.0;
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
  curandState *state, double *vMF_angle, int vMF_n)
{
	int gtid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gtid < N)
	{
		double4 xa = d_x[gtid];
		double4 na = d_n[gtid];

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
        int indx = floor(curand_uniform(&localState) * vMF_n); //number between 0 and vMF_n
        state[gtid] = localState;

        angle = vMF_angle[indx];

        //printf("angle = %f", angle);
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

        //time step:
    		x_new.x = xa.x + na.x * dt + Gamma_inverse_x_b.x * dt;
    		x_new.y = xa.y + na.y * dt + Gamma_inverse_x_b.y * dt;
    		x_new.z = 0.0;
    		x_new.w = la;

        //orientation dynamics
        double3 n_b;

    		n_b.x = - epsilon_r * inverse_Pe_R * inverse_rotation * dUbdy_dna.x;
    		n_b.y = - epsilon_r * inverse_Pe_R * inverse_rotation * dUbdy_dna.y;
    		n_b.z = - epsilon_r * inverse_Pe_R * inverse_rotation * dUbdy_dna.z;

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

        n_new.x = na.x + ori_proj_n_b.x * dt;
        n_new.y = na.y + ori_proj_n_b.y * dt;
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
  double3& W_hat, double& C, double& L,
  double& epsilon_r, double& sigma_bdy,
  double& inverse_Pe_T, double& inverse_Pe_parallel, double& inverse_Pe_perp, double& inverse_Pe_R,
  double& delta_run, double& delta_tumble, int& kappa, int& vMF_n)
{
	ifstream infile(filename);
	if (infile.fail())
    {
        cout<<"\nSimulation parameters input file opening failed.\n";
        exit(1);
    }

  int number_inputs = 24;
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
  W_hat.x = input_vec[++i];
  W_hat.y = input_vec[++i];
  W_hat.z = input_vec[++i];
  C = input_vec[++i];
  L = input_vec[++i];
  epsilon_r = input_vec[++i];
  sigma_bdy = input_vec[++i];
  inverse_Pe_T = input_vec[++i];
  inverse_Pe_parallel = input_vec[++i];
  inverse_Pe_perp = input_vec[++i];
  inverse_Pe_R = input_vec[++i];
  delta_run = input_vec[++i];
  delta_tumble = input_vec[++i];
  kappa = int(input_vec[++i]);
  vMF_n = int(input_vec[++i]);

	cout << "\nSimulation parameters loaded\n";
}

void load_textfile_vMF(char filename[], double angle[], int num_coords)
{
	int row_indx = 0;

	ifstream infile(filename);
	if (infile.fail())
    {
        cout << "\nangle input file opening failed\n";
        exit(1);
    }

	while (row_indx < num_coords)
	{
		infile >> angle[row_indx];
		row_indx++;
	}
	cout << "\nangle data loaded\n\n";
}

void initial_loading(double4 x[], double4 n[], int N, double C, double L,
  double l, double d, double t_run[], double t_tumble[], double delta_run, double simulation_time)
{
  double factorC = 1.0;
  double factorL = 1.0;

  double factorLminus1 = 1.0 - factorL;

  double xmin = 0.0 + 0.5 * factorLminus1 * L;
  double xmax = L - 0.5 * factorLminus1 * L;

  uniform_real_distribution<double> uniform_x(xmin, xmax);
  uniform_real_distribution<double> uniform_y(-factorC * C, factorC * C);
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
  double3 W_hat, double C, double L,
  double epsilon_r, double sigma_bdy,
  double inverse_Pe_T, double inverse_Pe_parallel, double inverse_Pe_perp, double inverse_Pe_R,
  double delta_run, double delta_tumble, int kappa, int vMF_n)
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

  string headers("sim_num, case_num, dt, time_save, start_time, final_time, N, l, d, W_hat_1, W_hat_2, W_hat_3, C, L, epsilon_r, sigma_bdy, inverse_Pe_T, inverse_Pe_parallel, inverse_Pe_perp, inverse_Pe_R, delta_run, delta_tumble, kappa, vMF_n");

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
       << W_hat.x << ", "
       << W_hat.y << ", "
       << W_hat.z << ", "
       << C << ", "
       << L << ", "
       << epsilon_r << ", "
       << sigma_bdy << ", "
       << inverse_Pe_T << ", "
       << inverse_Pe_parallel << ", "
       << inverse_Pe_perp << ", "
       << inverse_Pe_R << ", "
       << delta_run << ", "
       << delta_tumble << ", "
       << kappa << ", "
       << vMF_n << endl;

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

    double3 W_hat;  //wall normal
    double C;       //wall surface displacement from origin
    double L;       //wall length

    double epsilon_r;
	  double sigma_bdy;				 //range parameter for bacteria-wall steric repulsion

    double inverse_Pe_T;
    double inverse_Pe_parallel;
    double inverse_Pe_perp;
    double inverse_Pe_R;

	  double delta_run;					//run time
    double delta_tumble;      //tumble time
	  int kappa;		//kappa for vonMisesFisher distribution
	  int vMF_n;		//number of vMF numbers

    load_textfile_sim_parameters( "bacteria_surface_input.txt",
    sim_num, case_num,
    dt, time_save, start_time, final_time,
    N, l, d,
    W_hat, C, L,
    epsilon_r, sigma_bdy,
    inverse_Pe_T, inverse_Pe_parallel, inverse_Pe_perp, inverse_Pe_R,
    delta_run, delta_tumble, kappa, vMF_n);

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
         << "W_hat = " << "< " << W_hat.x << ", " << W_hat.y << ", " << W_hat.z << ">" << endl
         << "C = " << C << endl
         << "L = " << L << endl
         << "epsilon_r = " << epsilon_r << endl
         << "sigma_bdy = " << sigma_bdy << endl
         << "inverse_Pe_T = " << inverse_Pe_T << endl
         << "inverse_Pe_parallel = " << inverse_Pe_parallel << endl
         << "inverse_Pe_perp = " << inverse_Pe_perp << endl
         << "inverse_Pe_R = " << inverse_Pe_R << endl
         << "delta_run = "<< delta_run << endl
         << "delta_tumble = " << delta_tumble << endl
         << "kappa = " << kappa << endl
         << "vMF_n = " << vMF_n << endl
         << "================"<<endl;
    cout.precision(15);

    print_to_file_input(sim_num, case_num, dt, time_save, start_time, final_time,
      N, l, d, W_hat, C, L, epsilon_r, sigma_bdy,
      inverse_Pe_T, inverse_Pe_parallel, inverse_Pe_perp, inverse_Pe_R, delta_run, delta_tumble, kappa, vMF_n);
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

    //von Mises-Fisher:
    double vMF_angle[vMF_n];
    char vMF_filename[50];
    sprintf(vMF_filename, "vonMisesFisher2D_kappa%d_n%d.txt", kappa, vMF_n);
    //-----End: declaring derived simulation parameters-----

    //-----Start: INITIALIZING-----
    //-----Start: load vMF vectors-----
    load_textfile_vMF(vMF_filename, vMF_angle, vMF_n);
    //-----End: load vMF vectors-----

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
    double *d_vMF_angle;

		// allocate GPU memory
		cudaMalloc((void**) &d_x, N * sizeof(double4));
		cudaMalloc((void**) &d_n, N * sizeof(double4));
		cudaMalloc((void**) &d_dUbdy_dxa, N * sizeof(double3));
		cudaMalloc((void**) &d_dUbdy_dna, N * sizeof(double3));
		cudaMalloc((void**) &d_t_run, N * sizeof(double));
    cudaMalloc((void**) &d_t_tumble, N * sizeof(double));
    cudaMalloc((void**) &d_tumble_flag, N * sizeof(int));
    cudaMalloc((void **)&d_random_numbers_noise, N * sizeof(float4));
    cudaMalloc((void**)&d_vMF_angle, vMF_n * sizeof(double));

		// transfer the array to the GPU
		cudaMemcpy(d_x, x, N * sizeof(double4), cudaMemcpyHostToDevice);
		cudaMemcpy(d_n, n, N * sizeof(double4), cudaMemcpyHostToDevice);
		cudaMemcpy(d_t_run, t_run, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t_tumble, t_tumble, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tumble_flag, tumble_flag, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vMF_angle, vMF_angle, vMF_n * sizeof(double), cudaMemcpyHostToDevice);

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

      //-----Start: boundary interactions-----
			calculate_BodyWallInteraction<<< num_BLOCKS, num_THREADS >>>(d_dUbdy_dxa,
        d_dUbdy_dna, d_x, d_n, W_hat,
        C, sigma_bdy, N);
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
        d_CurandStates, d_vMF_angle, vMF_n);
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
