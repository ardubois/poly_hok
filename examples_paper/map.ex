require PolyHok

PolyHok.defmodule PMap do
  defk map_ker(a1,a2,size,f) do
    var id int = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(index,size,stride) do
          a2[id] = f(a1[id])
    end
  end
  defd inc(x) do
    x+1
  end
  def map(v1, f) do
    {l,c} = Hok.get_shape_gnx(v1)
    type = Hok.get_type_gnx(v1)
    size = l*c

    threadsPerBlock = 128;
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    result_gpu =Hok.new_gnx(l,c,type)



    Hok.spawn_jit(&PMap.map_ske/4,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[v1,result_gpu,size, f])
    result_gpu

  end

end

#a = Hok.hok (fn x,y -> x+y end)
#IO.inspect a
#raise "hell"

tensor1 = Nx.tensor([[1,2,3,4]],type: {:s, 32})
tensor2 = Nx.tensor([[1,2,3,4]],type: {:f, 32})
tensor3 = Nx.tensor([[1,2,3,4]],type: {:f, 64})
gtensor1 = Hok.new_gnx(tensor1)
gtensor2 = Hok.new_gnx(tensor2)
gtensor3 = Hok.new_gnx(tensor3)

func = Hok.hok fn (x) -> x + 1 end

#PMap.map(gtensor,&PMap.inc/1)

prev = System.monotonic_time()

gtensor1
    |> PMap.map(func)
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
