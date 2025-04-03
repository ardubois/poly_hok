require PolyHok
#PolyHok.defmodule SkeKernels do
#
# defk map_step_2_para_no_resp_kernel(d_array,  step, par1, par2,size,f) do
#    globalId  = blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x
#    id  = step * globalId
#    #f(id,id)
#    if (globalId < size) do
#      f(d_array+id,par1,par2)
#    end
#  end
#
#end

PolyHok.defmodule Ske do
  #defmacro __using__(_opts) do
  #     IO.puts "You are USIng!"
  #    end
  @defaults %{coord: false, return: true, dim: :one}
  def map(a,b,c,options \\[])
  def map({:nx, type, shape, name , ref}, func, [par1,par2], options )do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
  case dim do
    :one ->   if (not coord && not return )do
                   map_2_para_no_resp({:nx, type, shape, name , ref},  par1, par2, func)
              end

              if (not coord && return) do
                # map_2_para
              end
     :two ->  if (coord && not return) do
                    map_coord_2D_2_para_no_resp({:nx, type, shape, name , ref}, par1, par2, func)
              end


  end
end
  def map({:nx, type, shape, name , ref}, func, [par1], options )do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
  case dim do
    :one ->   if (not coord && not return )do
                   # map_1_para_no_resp({:nx, type, shape, name , ref},  par1, func)
              end

              if (not coord && return) do
                # map_2_para
              end
     :two ->  if (coord && not return) do
                    map_coord_2D_1_para_no_resp({:nx, type, shape, name , ref}, par1,  func)
              end


  end
  end
  def map({:nx, type, shape, name , ref}, {:nx, type, shape, name , ref}, func, options) do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)

    if (coord || not return || dim == :two) do
      raise "The only options for a map2 are: #{IO.inspect {coord: false, return: true, dim: :one}}"
    else
      map2({:nx, type, shape, name , ref}, {:nx, type, shape, name , ref}, func)
    end
  end
  def map(input, f) do
    shape = PolyHok.get_shape(input)
    type = PolyHok.get_type(input)
    result_gpu = PolyHok.new_gnx(shape,type)
    size = Tuple.product(shape)
    threadsPerBlock = 128;
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    PolyHok.spawn(&Ske.map_ker/4,
              {numberOfBlocks,1,1},
              {threadsPerBlock,1,1},
              [input,result_gpu,size, f])
    result_gpu
  end
  defk map2_kernel(a1,a2,a3,size,f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    if(id < size) do
      a3[id] = f(a1[id],a2[id])
    end
  end
  def map2(t1,t2,func) do

    {l,c} = PolyHok.get_shape_gnx(t1)
    type = PolyHok.get_type_gnx(t2)
     size = l*c
     result_gpu = PolyHok.new_gnx(l,c, type)

      threadsPerBlock = 256;
      numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

      PolyHok.spawn(&Ske.map2_kernel/5,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[t1,t2,result_gpu,size,func])


      result_gpu
  end
  defk map_coord_2D_1_para_no_resp_kernel(d_array,  step, par1,sizex,sizey,f) do

    x = threadIdx.x + blockIdx.x * blockDim.x
    y = threadIdx.y + blockIdx.y * blockDim.y
    offset = x + y * blockDim.x * gridDim.x

     id  = step * offset
    #f(id,id)
    if (offset < (sizex*sizey)) do
      f(d_array+id,par1,x,y)
    end
  end
  def map_coord_2D_1_para_no_resp(d_array, par1, f) do

   {sizex,sizey,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end

    #IO.inspect {sizex,sizey,step}
    block_size = 16
    grid_rows = trunc ((sizex + block_size - 1) / block_size)
    grid_cols = trunc ((sizey + block_size - 1) / block_size)


    PolyHok.spawn(&Ske.map_coord_2D_1_para_no_resp_kernel/6,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,step,par1,sizex,sizey,f])
      d_array
  end
  defk map_coord_2D_2_para_no_resp_kernel(d_array,  step, par1, par2,sizex,sizey,f) do

    x = threadIdx.x + blockIdx.x * blockDim.x
    y = threadIdx.y + blockIdx.y * blockDim.y
    offset = x + y * blockDim.x * gridDim.x

     id  = step * offset
    #f(id,id)
    if (offset < (sizex*sizey)) do
      f(d_array+id,par1,par2,x,y)
    end
  end
  def map_coord_2D_2_para_no_resp(d_array, par1, par2, f) do

   {sizex,sizey,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end

    #IO.inspect {sizex,sizey,step}
    block_size = 16
    grid_rows = trunc ((sizex + block_size - 1) / block_size)
    grid_cols = trunc ((sizey + block_size - 1) / block_size)


    PolyHok.spawn(&Ske.map_coord_2D_2_para_no_resp_kernel/7,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,step,par1,par2,sizex,sizey,f])
      d_array
  end
  def map_2_para_no_resp(d_array,  par1, par2, f) do
      block_size =  128;
      {l,step} = PolyHok.get_shape_gnx(d_array)
      size = l*step
      nBlocks = floor ((size + block_size - 1) / block_size)

      PolyHok.spawn(&Ske.map_step_2_para_no_resp_kernel/6,{nBlocks,1,1},{block_size,1,1},[d_array,step,par1,par2,l,f])
      d_array
  end
  defk map_step_2_para_no_resp_kernel(d_array,  step, par1, par2,size,f) do
        globalId  = blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x
        id  = step * globalId
        #f(id,id)
        if (globalId < size) do
          f(d_array+id,par1,par2)
        end
  end
  defk map_ker(a1,a2,size,f) do
      index = blockIdx.x * blockDim.x + threadIdx.x
      stride = blockDim.x * gridDim.x

      for i in range(index,size,stride) do
            a2[i] = f(a1[i])
      end
    end
  end





#### kernels
