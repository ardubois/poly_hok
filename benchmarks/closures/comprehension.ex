require PolyHok
use Comp

[arg] = System.argv()
size = String.to_integer(arg)

host_a = Enum.to_list(1..size) |> Nx.tensor(type: :f64)
host_b = Enum.to_list(1..size) |> Nx.tensor(type: :f64)

prev = System.monotonic_time()

a = PolyHok.new_gnx(host_a)
b = PolyHok.new_gnx(host_b)

_result = (Comp.gpu_for i <- 0..size, do:  2 * a[i] + b[i]) |> PolyHok.get_gnx

next = System.monotonic_time()
IO.puts "PolyHok\t#{size}\t#{System.convert_time_unit(next-prev,:native,:millisecond)} "
