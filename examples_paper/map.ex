require PolyHok

PolyHok.defmodule PMap do
  defk map_ker(a1,a2,size,f) do
    index = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(index,size,stride) do
          a2[i] = f(a1[i])
    end
  end
  defd inc(x) do
    x+1
  end
  def map(input, f) do
    shape = PolyHok.get_shape(input)
    type = PolyHok.get_type(input)
    result_gpu = PolyHok.new_gnx(shape,type)

    size = Tuple.product(shape)
    threadsPerBlock = 128;
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    PolyHok.spawn(&PMap.map_ker/4,
              {numberOfBlocks,1,1},
              {threadsPerBlock,1,1},
              [input,result_gpu,size, f])
    result_gpu
  end
end

#a = Hok.hok (fn x,y -> x+y end)
#IO.inspect a
#raise "hell"

arr1 = Nx.tensor([[1,2,3,4]],type: {:s, 32})
arr2 = Nx.tensor([[1,2,3,4]],type: {:f, 32})
arr3 = Nx.tensor([[1,2,3,4]],type: {:f, 64})

host_res1 = arr1
    |> PolyHok.new_gnx
    |> PMap.map(&PMap.inc/1)
    |> PolyHok.get_gnx

host_res2 = arr2
    |> PolyHok.new_gnx
    |> PMap.map(&PMap.inc/1)
    |> PolyHok.get_gnx

host_res3 = arr3
    |> PolyHok.new_gnx
    |> PMap.map(PolyHok.phok fn (x) -> x + 1 end)
    |> PolyHok.get_gnx


gtensor1
    |> PMap.map()
    |> Hok.get_gnx
    |> IO.inspect

gtensor2
    |> PMap.map(func)
    |> Hok.get_gnx
    |> IO.inspect

gtensor3
    |> PMap.map(func)
    |> Hok.get_gnx
    |> IO.inspect

next = System.monotonic_time()
IO.puts "Hok\t\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"
