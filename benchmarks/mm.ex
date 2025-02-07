
require Hok

Hok.defmodule_jit MM do

defk map2xy2D_kernel(arr1,arr2,par, resp,size,f) do
  row  = blockIdx.y * blockDim.y + threadIdx.y
  col = blockIdx.x * blockDim.x + threadIdx.x

  if(col < size && row < size) do
    resp[row * size + col] = f(arr1,arr2,par,row,col)
  end
end
def map2xy2D1p(arr1,arr2,par,resp,size,f) do
  block_size = 16
  grid_rows = trunc ((size + block_size - 1) / block_size)
  grid_cols = trunc ((size + block_size - 1) / block_size)

  Hok.spawn_jit(&MM.map2xy2D_kernel/6,{grid_cols,grid_rows,1},{block_size,block_size,1},[arr1,arr2,par,resp,size,f])
end
def comp2xy2D1p(arr1,arr2,par,size1,size2,f) do


    result_gpu = Hok.new_gnx(size1,size2,Hok.get_array_type(arr1))
    arr1_gpu = Hok.new_gnx(arr1)
    arr2_gpu = Hok.new_gnx(arr2)

    MM.map2xy2D1p(arr1_gpu, arr2_gpu,par, result_gpu, size1,f)

    r_gpu = Hok.get_gnx(result_gpu)
    r_gpu
end
end

[arg] = System.argv()

m = String.to_integer(arg)

#vet1 = Nx.iota({m,m}, type: :f32)
#vet2 = Nx.iota({m,m}, type: :f32)

#{mat1,_} = Nx.Random.uniform(Nx.Random.key(1), shape: {m, m}, type: :f32)
#{mat2,_} = Nx.Random.uniform(Nx.Random.key(1), shape: {m, m}, type: :f32)

#mat1 = Matrex.new(1, m*m, fn -> :rand.uniform(1000) end)
#mat2 = Matrex.new(1, m*m, fn -> :rand.uniform(1000) end)

mat1 = Hok.new_nx_from_function(1,m*m,{:f,64},fn -> :rand.uniform(1000) end )
mat2 = Hok.new_nx_from_function(1,m*m,{:f,64},fn -> :rand.uniform(1000) end)

prev = System.monotonic_time()



_result = Hok.gpufor x <- 0..m, y <- 0..m, mat1, mat2,m do
            sum = 0
            for i in range(0,m,1) do
                  sum = sum + mat1[x * m + i] * mat2[i * m + y]
            end
            sum
          end

next = System.monotonic_time()

IO.puts "Hok\t#{m}\t#{System.convert_time_unit(next-prev,:native,:millisecond)} "

#IO.inspect result

#IO.inspect Nx.sum(result)

#Hok.null(mat1)
#Hok.null(mat2)
#m1 = Matrex.reshape(mat1,m,m)
#m2 = Matrex.reshape(mat2,m,m)
#res_cpu = Matrex.dot(m1,m2)
#IO.inspect Matrex.sum(res_cpu)
#IO.inspect Matrex.sum(result)
