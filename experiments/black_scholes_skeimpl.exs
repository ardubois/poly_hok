require PolyHok

defmodule BlackScholesSkeExperiment do
  require PolyHok
  use Ske

  @riskfree 0.02
  @volatility 0.30
  @opt_n 100_000
  @l1_tolerance 1.0e-6

  def run do
    :rand.seed(:exsplus, {5_347, 5_347, 5_347})

    stock = random_tensor(@opt_n, 5.0, 30.0)
    strike = random_tensor(@opt_n, 1.0, 100.0)
    years = random_tensor(@opt_n, 0.25, 10.0)

    stock_gpu = PolyHok.new_gnx(stock)
    strike_gpu = PolyHok.new_gnx(strike)
    years_gpu = PolyHok.new_gnx(years)

    call_result_gpu = run_ske_map3(stock_gpu, strike_gpu, years_gpu)
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
    IO.puts("Max absolute error: " <> :erlang.float_to_binary(max_delta, [{:scientific, 6}]))
    IO.puts("[BlackScholes] - Test Summary")

    if l1norm > @l1_tolerance do
      IO.puts("Test failed!")
    else
      IO.puts("Test passed.")
    end
  end

  defp run_ske_map3(stock_gpu, strike_gpu, years_gpu) do
    Ske.map3(
      stock_gpu,
      strike_gpu,
      years_gpu,
      PolyHok.phok(fn s, x, t ->
        a1 = 0.31938153
        a2 = -0.356563782
        a3 = 1.781477937
        a4 = -1.821255978
        a5 = 1.330274429
        rsqrt_2pi = 0.3989422804014327
        p = 0.2316419
        riskfree = 0.02
        volatility = 0.30

        sqrt_t = sqrtf(t)
        d1 = (logf(s / x) + (riskfree + (volatility * volatility) / 2.0) * t) / (volatility * sqrt_t)
        d2 = d1 - volatility * sqrt_t

        ax1 = fabsf(d1)
        k1 = 1.0 / (1.0 + p * ax1)
        poly1 = (((((a5 * k1 + a4) * k1 + a3) * k1 + a2) * k1 + a1) * k1)
        phi1 = rsqrt_2pi * expf(-0.5 * ax1 * ax1)
        cnd1 = 1.0 - phi1 * poly1

        if d1 < 0.0 do
          cnd1 = 1.0 - cnd1
        end

        ax2 = fabsf(d2)
        k2 = 1.0 / (1.0 + p * ax2)
        poly2 = (((((a5 * k2 + a4) * k2 + a3) * k2 + a2) * k2 + a1) * k2)
        phi2 = rsqrt_2pi * expf(-0.5 * ax2 * ax2)
        cnd2 = 1.0 - phi2 * poly2

        if d2 < 0.0 do
          cnd2 = 1.0 - cnd2
        end

        return s * cnd1 - x * expf(-riskfree * t) * cnd2
      end)
    )
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

BlackScholesSkeExperiment.run()
