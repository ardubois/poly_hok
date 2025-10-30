require PolyHok
use Comp

[arg] = System.argv()
size = String.to_integer(arg)

cpu_prev = System.monotonic_time()

host_a = Enum.to_list(1..size) |> Nx.tensor(type: :f64)
host_b = Enum.to_list(1..size) |> Nx.tensor(type: :f64)

cpu_next = System.monotonic_time()
IO.puts "Tensor Creation\t#{size}\t#{System.convert_time_unit(cpu_next-cpu_prev,:native,:millisecond)} "

prev = System.monotonic_time()

a = PolyHok.new_gnx(host_a)
b = PolyHok.new_gnx(host_b)

result = (Comp.gpu_for i <- 0..size, do:  2 * a[i] + b[i])

next = System.monotonic_time()
IO.puts "GPU For\t#{size}\t#{System.convert_time_unit(next-prev,:native,:millisecond)} "

cpu_prev = System.monotonic_time()

_result_cpu = result |> PolyHok.get_gnx

cpu_next = System.monotonic_time()
IO.puts "Get Tensor\t#{size}\t#{System.convert_time_unit(cpu_next-cpu_prev,:native,:millisecond)} "

# IO.puts "PolyHok\t#{size}\t#{System.convert_time_unit(next-prev,:native,:millisecond)} "
