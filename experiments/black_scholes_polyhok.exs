require PolyHok

PolyHok.defmodule BlackScholesExp do
  defd norm_cdf(x) do
    a1 = 0.31938153
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    rsqrt_2pi = 0.3989422804014327
    p = 0.2316419

    ax = fabsf(x)
    k = 1.0 / (1.0 + p * ax)
    poly = (((((a5 * k + a4) * k + a3) * k + a2) * k + a1) * k)
    phi = rsqrt_2pi * expf(-0.5 * ax * ax)
    cnd = 1.0 - phi * poly

    if x < 0.0 do
      cnd = 1.0 - cnd
    end

    return cnd
  end

  defd blackscholes_body(s, x, t, r, v) do
    sqrt_t = sqrtf(t)
    d1 = (logf(s / x) + (r + (v * v) / 2.0) * t) / (v * sqrt_t)
    d2 = d1 - v * sqrt_t
    return s * norm_cdf(d1) - x * expf(-r * t) * norm_cdf(d2)
  end

  defk black_scholes(call_result, stock_price, option_strike, option_years, riskfree, volatility, n) do
    opt = threadIdx.x + blockIdx.x * blockDim.x

    if opt < n do
      call_result[opt] =
        blackscholes_body(stock_price[opt], option_strike[opt], option_years[opt], riskfree, volatility)
    end
  end
end

defmodule BlackScholesExperiment do
  @riskfree 0.02
  @volatility 0.30
  @opt_n 100_000
  @threads_per_block 256
  @l1_tolerance 1.0e-6

  def run do
    :rand.seed(:exsplus, {5_347, 5_347, 5_347})

    stock = random_tensor(@opt_n, 5.0, 30.0)
    strike = random_tensor(@opt_n, 1.0, 100.0)
    years = random_tensor(@opt_n, 0.25, 10.0)

    stock_gpu = PolyHok.new_gnx(stock)
    strike_gpu = PolyHok.new_gnx(strike)
    years_gpu = PolyHok.new_gnx(years)

    call_result_gpu = run_kernel(stock_gpu, strike_gpu, years_gpu, @riskfree, @volatility)
    call_result = call_result_gpu |> PolyHok.get_gnx() |> Nx.to_flat_list()

    stock_list = Nx.to_flat_list(stock)
    strike_list = Nx.to_flat_list(strike)
    years_list = Nx.to_flat_list(years)

    call_result_cpu =
      stock_list
      |> Enum.zip(strike_list)
      |> Enum.zip(years_list)
      |> Enum.map(fn {{s, x}, t} -> black_scholes_body_cpu(s, x, t, @riskfree, @volatility) end)

    {l1norm, max_delta} = compare(call_result_cpu, call_result)

    IO.puts("Checking the results...")
    IO.puts("Comparing the results...")
    IO.puts("L1 norm: " <> :erlang.float_to_binary(l1norm, [{:scientific, 6}]))
    # IO.puts("L1 norm: #{:io_lib.format("~.6E", [l1norm]) |> IO.iodata_to_binary()}")
    IO.puts("Max absolute error: " <> :erlang.float_to_binary(max_delta, [{:scientific, 6}]))
    # IO.puts("Max absolute error: #{:io_lib.format("~.6E", [max_delta]) |> IO.iodata_to_binary()}")
    IO.puts("[BlackScholes] - Test Summary")

    if l1norm > @l1_tolerance do
      IO.puts("Test failed!")
    else
      IO.puts("Test passed.")
    end
  end

  defp run_kernel(stock_gpu, strike_gpu, years_gpu, riskfree, volatility) do
    shape = PolyHok.get_shape_gnx(stock_gpu)
    size = Tuple.product(shape)

    if PolyHok.get_shape_gnx(strike_gpu) != shape or PolyHok.get_shape_gnx(years_gpu) != shape do
      raise "run_kernel: input shapes must match"
    end

    unless PolyHok.get_type_gnx(stock_gpu) == {:f, 32} and PolyHok.get_type_gnx(strike_gpu) == {:f, 32} and
             PolyHok.get_type_gnx(years_gpu) == {:f, 32} do
      raise "run_kernel: only {:f, 32} tensors are supported"
    end

    call_result_gpu = PolyHok.new_gnx(shape, {:f, 32})
    blocks = div(size + @threads_per_block - 1, @threads_per_block)

    PolyHok.spawn(
      &BlackScholesExp.black_scholes/7,
      {blocks, 1, 1},
      {@threads_per_block, 1, 1},
      [call_result_gpu, stock_gpu, strike_gpu, years_gpu, riskfree, volatility, size]
    )

    call_result_gpu
  end

  defp random_tensor(n, low, high) do
    vals =
      for _ <- 1..n do
        t = :rand.uniform()
        (1.0 - t) * low + t * high
      end

    Nx.tensor(vals, type: {:f, 32})
  end

  defp cnd(d) do
    a1 = 0.31938153
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    rsqrt_2pi = 0.3989422804014327

    k = 1.0 / (1.0 + 0.2316419 * abs(d))

    cnd =
      rsqrt_2pi * :math.exp(-0.5 * d * d) *
        (k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5)))))

    if d > 0.0, do: 1.0 - cnd, else: cnd
  end

  defp black_scholes_body_cpu(s, x, t, r, v) do
    sqrt_t = :math.sqrt(t)
    d1 = (:math.log(s / x) + (r + 0.5 * v * v) * t) / (v * sqrt_t)
    d2 = d1 - v * sqrt_t
    s * cnd(d1) - x * :math.exp(-r * t) * cnd(d2)
  end

  defp compare(ref_values, gpu_values) do
    {sum_delta, sum_ref, max_delta} =
      Enum.zip(ref_values, gpu_values)
      |> Enum.reduce({0.0, 0.0, 0.0}, fn {ref, gpu}, {acc_delta, acc_ref, acc_max} ->
        delta = abs(ref - gpu)
        {acc_delta + delta, acc_ref + abs(ref), max(acc_max, delta)}
      end)

    l1norm = if sum_ref == 0.0, do: 0.0, else: sum_delta / sum_ref
    {l1norm, max_delta}
  end
end

BlackScholesExperiment.run()
