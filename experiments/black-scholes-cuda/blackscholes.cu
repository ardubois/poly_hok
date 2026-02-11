#include <cstdlib>
#include <math.h>
#include <stdio.h>

const int OPT_N = 4000000;
const int OPT_SZ = OPT_N * sizeof(float);
const float RISKFREE = 0.02f;
const float VOLATILITY = 0.30f;

/*
 *
 * CPU Implementation
 *
 *
 * */

double CND(double d) {
  const double A1 = 0.31938153;
  const double A2 = -0.356563782;
  const double A3 = 1.781477937;
  const double A4 = -1.821255978;
  const double A5 = 1.330274429;
  const double RSQRT2PI = 0.39894228040143267793994605993438;

  double K = 1.0 / (1.0 + 0.2316419 * fabs(d));

  double cnd = RSQRT2PI * exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

  if (d > 0)
    cnd = 1.0 - cnd;

  return cnd;
}

void BlackScholesBodyCPU(float &callResult, float &putResult,
                         float Sf, // Stock price
                         float Xf, // Option strike
                         float Tf, // Option years
                         float Rf, // Riskless rate
                         float Vf  // Volatility rate
) {
  double S = Sf, X = Xf, T = Tf, R = Rf, V = Vf;

  double sqrtT = sqrt(T);
  double d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
  double d2 = d1 - V * sqrtT;
  double CNDD1 = CND(d1);
  double CNDD2 = CND(d2);

  // Calculate Call and Put simultaneously
  double expRT = exp(-R * T);
  callResult = (float)(S * CNDD1 - X * expRT * CNDD2);
  putResult = (float)(X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1));
}

void BlackScholesCPU(float *h_CallResult, float *h_PutResult,
                     float *h_StockPrice, float *h_OptionStrike,
                     float *h_OptionYears, float Riskfree, float Volatility,
                     int optN) {
  for (int opt = 0; opt < optN; opt++)
    BlackScholesBodyCPU(h_CallResult[opt], h_PutResult[opt], h_StockPrice[opt],
                        h_OptionStrike[opt], h_OptionYears[opt], Riskfree,
                        Volatility);
}

float RandFloat(float low, float high) {
  float t = (float)rand() / (float)RAND_MAX;
  return (1.0f - t) * low + t * high;
}

__device__ inline float norm_cdf(float x) {

  const float a1 = 0.31938153f;
  const float a2 = -0.356563782f;
  const float a3 = 1.781477937f;
  const float a4 = -1.821255978f;
  const float a5 = 1.330274429f;
  const float rsqrt_2pi = 0.39894228040143267793994605993438f;
  const float p = 0.2316419f;

  float ax = fabsf(x);
  float k = 1.0f / (1.0f + p * ax);
  float poly = (((((a5 * k + a4) * k + a3)) * k + a2) * k + a1) * k;
  float phi = rsqrt_2pi * expf(-0.5f * ax * ax);
  float cnd = 1.0f - phi * poly;

  if (x < 0)
    cnd = 1.0f - cnd;
  return cnd;
}

/*
 * S -> Stock price
 * X -> Option strike
 * T -> Option years
 * R -> Riskless rate
 * V -> Volatility rate
 */
__device__ float blackscholes_body(float S, float X, float T, float r,
                                   float v) {

  float sqrtT = sqrtf(T);
  float d1 = (logf(S / X) + (r + (v * v) / 2.0f) * T) / (v * sqrtT);
  float d2 = d1 - v * sqrtT;
  return S * norm_cdf(d1) - X * __expf(-r * T) * norm_cdf(d2);
}

__global__ void black_scholes(float *call_result, float *stock_price,
                              float *option_strike, float *option_years,
                              float riskfree, float volatility) {

  const int opt = threadIdx.x + blockIdx.x * blockDim.x;
  call_result[opt] = blackscholes_body(stock_price[opt], option_strike[opt],
                                       option_years[opt], riskfree, volatility);
}

int main(int argc, char **argv) {

  float *call_result, *call_result_cpu, *stock_price, *option_strike,
      *option_years;
  float *d_call_result, *d_stock_price, *d_option_strike, *d_option_years;

  // Allocating CPU memory for options
  call_result = (float *)malloc(OPT_SZ);
  call_result_cpu = (float *)malloc(OPT_SZ);
  stock_price = (float *)malloc(OPT_SZ);
  option_strike = (float *)malloc(OPT_SZ);
  option_years = (float *)malloc(OPT_SZ);

  cudaMalloc((void **)&d_call_result, OPT_SZ);
  cudaMalloc((void **)&d_stock_price, OPT_SZ);
  cudaMalloc((void **)&d_option_strike, OPT_SZ);
  cudaMalloc((void **)&d_option_years, OPT_SZ);
  float *put_result_cpu = (float *)malloc(OPT_SZ);

  srand(5347);

  // Generate options set
  for (int i = 0; i < OPT_N; i++) {
    call_result[i] = 0.0;
    stock_price[i] = RandFloat(5.0f, 30.0f);
    option_strike[i] = RandFloat(1.0f, 100.0f);
    option_years[i] = RandFloat(0.25f, 10.0f);
  }
  // Copy options data to GPU memory for further processing

  cudaMemcpy(d_stock_price, stock_price, OPT_SZ, cudaMemcpyHostToDevice);
  cudaMemcpy(d_option_strike, option_strike, OPT_SZ, cudaMemcpyHostToDevice);
  cudaMemcpy(d_option_years, option_years, OPT_SZ, cudaMemcpyHostToDevice);

  int numThreads = 256;
  int numBlocks = (OPT_N + numThreads - 1) / numThreads;
  black_scholes<<<numBlocks, numThreads>>>(
      (float *)d_call_result, (float *)d_stock_price, (float *)d_option_strike,
      (float *)d_option_years, RISKFREE, VOLATILITY);

  cudaDeviceSynchronize();

  cudaMemcpy(call_result, d_call_result, OPT_SZ, cudaMemcpyDeviceToHost);

  printf("Checking the results...\n");

  BlackScholesCPU(call_result_cpu, put_result_cpu, stock_price, option_strike,
                  option_years, RISKFREE, VOLATILITY, OPT_N);

  double delta, ref, sum_delta, sum_ref, max_delta, L1norm, gpuTime;
  printf("Comparing the results...\n");
  // Calculate max absolute difference and L1 distance
  // between CPU and GPU results
  sum_delta = 0;
  sum_ref = 0;
  max_delta = 0;

  for (int i = 0; i < OPT_N; i++) {

    ref = call_result_cpu[i];
    delta = fabs(call_result_cpu[i] - call_result[i]);
    if (delta > max_delta) {
      max_delta = delta;
    }
    sum_delta += delta;
    sum_ref += fabs(ref);
  }
  L1norm = sum_delta / sum_ref;
  printf("L1 norm: %E\n", L1norm);
  printf("Max absolute error: %E\n\n", max_delta);
  printf("\n[BlackScholes] - Test Summary\n");

    if (L1norm > 1e-6)
    {
        printf("Test failed!\n");
    }

  cudaFree(d_option_years);
  cudaFree(d_option_strike);
  cudaFree(d_stock_price);
  cudaFree(d_call_result);

  free(call_result);
  free(stock_price);
  free(option_strike);
  free(option_years);
  return 0;
}
