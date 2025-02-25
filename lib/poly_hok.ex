defmodule PolyHok do
  @on_load :load_nifs
  def load_nifs do
      :erlang.load_nif('./priv/gpu_nifs', 0)
      #IO.puts("ok")
  end

  defmacro phok({:fn, aa, [{:->, bb , [para,body]}] }) do
   # IO.inspect "body: #{inspect body}"
    body =  PolyHok.CudaBackend.add_return(body)
    name = "anon_" <> PolyHok.CudaBackend.gen_lambda_name()
    function = {:fn, aa, [{:->, bb , [para,body]}] }
    resp =  quote(do: {:anon , unquote(name),unquote(Macro.escape function)})
  #  resp =  quote(do: {:anon , unquote(name),unquote({:fn, aa, [{:->, bb , [para,body]}] })})
     resp
    #IO.inspect function
    #raise "hell"
    #{fname,type} = PolyHok.CudaBackend.gen_lambda("Elixir.App",function)
    #result = quote do: PolyHok.load_lambda_compilation(unquote("Elixir.App"), unquote(fname), unquote(type))
    #result
    #IO.inspect result
    #raise "hell"

  end

  defmacro gpufor({:<-, _ ,[var,tensor]},do: b)  do
      quote do: Comp.comp(unquote(tensor), PolyHok.phok (fn (unquote(var)) -> (unquote b) end))
  end

  defmacro gpufor({:<-,_, [var1, {:..,_, [_b1, e1]}]}, arr1, arr2,do: body) do
       r=      quote do: Comp.comp_xy_2arrays(unquote(arr1), unquote(arr2), unquote(e1),
                                          PolyHok.phok (fn (unquote(arr1),
                                                       unquote(arr2),
                                                       unquote(var1)) -> (unquote body) end))
      #IO.inspect r
       #raise "hell"
       r
  end

  defmacro gpufor({:<-,_, [var1, {:..,_, [_b1, e1]}]}, {:<-,_, [var2, {:..,_, [_b2, e2]}]},arr1,arr2, par3, do: body) do

       #IO.inspect "Aqui"
       r=      quote do: MM.comp2xy2D1p(unquote(arr1), unquote(arr2), unquote(par3), unquote(e1), unquote(e2),
                                          PolyHok.phok (fn (unquote(arr1),
                                                       unquote(arr2),
                                                       unquote(par3),
                                                       unquote(var1),
                                                       unquote(var2)) -> (unquote body) end))
       #IO.inspect r
       #raise "hell"
       r
  end


  defmacro defmodule_st(header,do: body) do
    #IO.inspect header
    #IO.inspect body
    {:__aliases__, _, [module_name]} = header


    code = PolyHok.CudaBackend.compile_module(module_name,body)

    file = File.open!("c_src/Elixir.#{module_name}.cu", [:write])
    IO.write(file, "#include \"erl_nif.h\"\n\n" <> code)
    File.close(file)
    {result, errcode} = System.cmd("nvcc",
        [ "--shared",
          "--compiler-options",
          "'-fPIC'",
          "-o",
          "priv/Elixir.#{module_name}.so",
          "c_src/Elixir.#{module_name}.cu"
  ], stderr_to_stdout: true)



  if ((errcode == 1) || (errcode ==2)) do raise "Error when compiling .cu file generated by PolyHok:\n#{result}" end

    ast_new_module = PolyHok.CudaBackend.gen_new_module(header,body)
    #IO.inspect ast_new_module
    ast_new_module


    #quote do: IO.puts "ok"
  end

  defmacro defmodule(header,do: body) do
  #  IO.inspect header
    #IO.inspect body
    {:__aliases__, _, [module_name]} = header


    JIT.process_module(module_name,body)

    ast_new_module = PolyHok.CudaBackend.gen_new_module(header,body)
    #IO.inspect ast_new_module
    ast_new_module


    #quote do: IO.puts "ok"
  end

  defmacro include(inc_list) do
    #IO.inspect inc_list
    includes = inc_list
                |> Enum.map(fn {_,_,[module]} -> to_string(module) end)

    file = File.open!("c_src/Elixir.App.cu", [:write])
    IO.write(file, "#include \"erl_nif.h\"\n\n")
    Enum.map(includes, fn module ->   code = File.read!("c_src/Elixir.#{module}.cu")
                                          |>  String.split("\n")
                                          |>  Enum.drop(1)
                                          |> Enum.join("\n")
                                      IO.write(file, code)  end)
    File.close(file)

    {result, errcode} = System.cmd("nvcc",
        [ "--shared",
          "--compiler-options",
          "'-fPIC'",
          "-o",
          "priv/Elixir.App.so",
          "c_src/Elixir.App.cu"
  ], stderr_to_stdout: true)


  if ((errcode == 1) || (errcode ==2)) do raise "Error when compiling .cu file generated by PolyHok:\n#{result}" end


  end

  #####################
  #####
  ##### Legacy code:  gptype macro ##########
  #####
  ##########################

  defmacro deft({func,_,[type]}) do

    if (nil == Process.whereis(:gptype_server)) do
      pid = spawn_link(fn -> gptype_server() end)
      Process.register(pid, :gptype_server)
    end
    send(:gptype_server,{:add_type, func,type_to_list(type)})
    #IO.inspect(type_to_list(type))
    quote do
    end
  end
  def gptype_server(), do: gptype_server_(Map.new())
  defp gptype_server_(map) do
    receive do
      {:add_type, fun, types}  -> map=Map.put(map,fun, types)
                              gptype_server_(map)
      {:get_type, pid,fun} -> type=Map.get(map,fun)
                              send(pid,{:type,fun,type})
                              gptype_server_(map)
      {:kill}               -> :dead
    end
  end
  defp type_to_list({:integer,_,_}), do: [:int]
  defp type_to_list({:unit,_,_}), do: [:unit]
  defp type_to_list({:float,_,_}), do: [:float]
  defp type_to_list({:gmatrex,_,_}), do: [:matrex]
  defp type_to_list([type]), do: [type_to_list(type)]
  defp type_to_list({:~>,_, [a1,a2]}), do: type_to_list(a1) ++ type_to_list(a2)
  defp type_to_list({x,_,_}), do: raise "Unknown type constructor #{x}"
  def is_typed?() do
    nil != Process.whereis(:gptype_server)
  end
  def get_type_kernel(fun_name) do
    send(:gptype_server,{:get_type, self(),fun_name})
    receive do
      {:type,fun,type} -> if fun == fun_name do
                                send(:gptype_server,{:kill})
                                type
                          else
                                raise "Asked for #{fun_name} got #{fun}"
                          end
      end

    end
    def get_type_fun(fun_name) do
      send(:gptype_server,{:get_type, self(),fun_name})
      receive do
        {:type,fun,type} -> if fun == fun_name do
                                  type
                            else
                                  raise "Asked for #{fun_name} got #{fun}"
                            end
        end

      end

  #############################################
  ###########
  #######   Legacy code Hok macro
  #######
  #####################




#defp gen_para(p,:matrex) do
#  "float *#{p}"
#end
#defp gen_para(p,:float) do
#  "float #{p}"
#end
#defp gen_para(p,:int) do
#  "int #{p}"
#end
#defp gen_para(p, list) when is_list(list) do
#  size = length(list)
#
#  {ret,type}=List.pop_at(list,size-1)
#
#  r="#{ret} (*#{p})(#{to_arg_list(type)})"
#  r
#
#end
#defp to_arg_list([t]) do
#  "#{t}"
#end
#defp to_arg_list([v|t]) do
#  "#{v}," <> to_arg_list(t)
#end

######################################
#########
########   GNX stuff
#######
##############################

def get_type_gnx({:nx, type, _shape, _name , _ref}) do
  type
end
def get_type({:nx, type, _shape, _name , _ref}) do
  type
end
def get_shape_gnx({:nx, _type, shape, _name , _ref}) do
  shape
end
def get_shape({:nx, _type, shape, _name , _ref}) do
  shape
end
def new_gnx(%Nx.Tensor{data: data, type: type, shape: shape, names: name}) do
  %Nx.BinaryBackend{ state: array} = data
 # IO.inspect name
 # raise "hell"
  {l,c} = case shape do
    {c} -> {1,c}
    {l,c} -> {l,c}
  end
  ref = case type do
     {:f,32} -> create_gpu_array_nx_nif(array,l,c,Kernel.to_charlist("float"))
     {:f,64} -> create_gpu_array_nx_nif(array,l,c,Kernel.to_charlist("double"))
     {:s,32} -> create_gpu_array_nx_nif(array,l,c,Kernel.to_charlist("int"))
     x -> raise "new_gnx: type #{x} not suported"
  end
  {:nx, type, shape, name , ref}
end
def new_gnx(%Matrex{data: matrix} = a) do
  #IO.puts "aqui!"
    <<l::unsigned-integer-little-32,c::unsigned-integer-little-32,z::binary>> = matrix
    ref = create_gpu_array_nx_nif(z,l,c,Kernel.to_charlist("float"))
  {:matrex, ref, Matrex.size(a)}
end
def new_gnx(l,c,type) do
 # IO.puts "aque"
  ref = case type do
    {:f,32} -> new_gpu_array_nif(l,c,Kernel.to_charlist("float"))
    {:f,64} -> new_gpu_array_nif(l,c,Kernel.to_charlist("double"))
    {:s,32} -> new_gpu_array_nif(l,c,Kernel.to_charlist("int"))
    x -> raise "new_gnx: type #{x} not suported"
 end

 {:nx, type, {l,c}, [nil,nil] , ref}
end
def new_gnx({l,c},type) do
  # IO.puts "aque"
   ref = case type do
     {:f,32} -> new_gpu_array_nif(l,c,Kernel.to_charlist("float"))
     {:f,64} -> new_gpu_array_nif(l,c,Kernel.to_charlist("double"))
     {:s,32} -> new_gpu_array_nif(l,c,Kernel.to_charlist("int"))
     x -> raise "new_gnx: type #{x} not suported"
  end

  {:nx, type, {l,c}, [nil,nil] , ref}
 end
def get_gnx({:matrex, ref, {rows,columns}}) do
  bin = get_gpu_array_nif(ref,rows,columns,Kernel.to_charlist("float"))
  array = <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32,bin::binary>>
  %Matrex{data: array}
end
def get_gnx({:nx, type, shape, name , ref}) do
  #IO.puts "aqui..."
  {l,c} = shape
  ref = case type do
    {:f,32} -> get_gpu_array_nif(ref,l,c,Kernel.to_charlist("float"))
    {:f,64} -> get_gpu_array_nif(ref,l,c,Kernel.to_charlist("double"))
    {:s,32} -> get_gpu_array_nif(ref,l,c,Kernel.to_charlist("int"))
    x -> raise "new_gnx: type #{x} not suported"
 end

  %Nx.Tensor{data: %Nx.BinaryBackend{ state: ref}, type: type, shape: shape, names: name}
end
#def get_gnx({:matrex, ref,{rows,cols}}) do
#  %Matrex{data: get_matrex_nif(ref,rows,cols)}
#end
def new_nx_from_function(l,c,type, fun) do
  size = l*c
  ref =case type do
    {:f,32} -> new_matrix_from_function_f(size-1,fun, <<fun.()::float-little-32>>)
    {:f,64} -> new_matrix_from_function_d(size-1,fun, <<fun.()::float-little-64>>)
    {:s,32} -> new_matrix_from_function_i(size-1,fun, <<fun.()::integer-little-32>>)
  end
   %Nx.Tensor{data: %Nx.BinaryBackend{ state: ref}, type: type, shape: {l,c}, names:  [nil,nil]}
end

#######################
defp new_matrix_from_function_d(0, _, accumulator), do: accumulator

  defp new_matrix_from_function_d(size, function, accumulator),
    do:
      new_matrix_from_function_d(
        size - 1,
        function,
        <<accumulator::binary, function.()::float-little-64>>
      )
defp new_matrix_from_function_i(0, _, accumulator), do: accumulator

  defp new_matrix_from_function_i(size, function, accumulator),
    do:
      new_matrix_from_function_i(
        size - 1,
        function,
        <<accumulator::binary, function.()::integer-little-32>>
      )
defp new_matrix_from_function_f(0, _, accumulator), do: accumulator

  defp new_matrix_from_function_f(size, function, accumulator),
    do:
      new_matrix_from_function_f(
        size - 1,
        function,
        <<accumulator::binary, function.()::float-little-32>>
      )
##############################
def new_gnx_fake(_size,type) do
  {:nx, type, :shape, :name, :ref}
end
def new_gnx_fake ((%Nx.Tensor{data: _data, type: type, shape: shape, names: name}) ) do
 # %Nx.BinaryBackend{ state: array} = data
  #{l,c} = shape
  #ref = case type do
   #  {:f,32} -> create_gpu_array_nx_nif(array,l,c,Kernel.to_charlist("float"))
   #  {:f,64} -> create_gpu_array_nx_nif(array,l,c,Kernel.to_charlist("double"))
   #  {:s,32} -> create_gpu_array_nx_nif(array,l,c,Kernel.to_charlist("int"))
   #  x -> raise "new_gmatrex: type #{x} not suported"
  #end
  {:nx, type, shape, name , :ref}
end
def get_array_type(%Nx.Tensor{data: _data, type: _type, shape: _shape, names: _name} = nx) do
  Nx.type(nx)
end
def get_array_type(%Matrex{data: _matrix}) do
  {:f,32}
end
def null(a) do
  a
end
def new_gpu_array_nif(_l,_c,_type) do
  raise "NIF new_gpu_array_nif/4 not implemented"
end
def get_gpu_array_nif(_matrex,_l,_c,_type) do
  raise "NIF get_gpu_array_nif/4 not implemented"
end
def create_gpu_array_nx_nif(_matrex,_l,_c,_type) do
  raise "NIF create_gpu_array_nx_nif/4 not implemented"
end




######################################
#########
########   GNX stuff OLD
#######
##############################

def get_type_gnx_({:nx, type, _shape, _name , _ref}) do
  type
end
def get_shape_gnx_({:nx, _type, shape, _name , _ref}) do
  shape
end
def new_gnx_((%Nx.Tensor{data: data, type: type, shape: shape, names: name}) ) do
  %Nx.BinaryBackend{ state: array} = data
 # IO.inspect name
 # raise "hell"
  {l,c} = case shape do
    {c} -> {1,c}
    {l,c} -> {l,c}
  end
  ref = case type do
     {:f,32} -> create_nx_ref_nif(array,l,c,Kernel.to_charlist("float"))
     {:f,64} -> create_nx_ref_nif(array,l,c,Kernel.to_charlist("double"))
     {:s,32} -> create_nx_ref_nif(array,l,c,Kernel.to_charlist("int"))
     x -> raise "new_gmatrex: type #{x} not suported"
  end
  {:nx, type, shape, name , ref}
end
def new_gnx_(l,c,type) do

  ref = case type do
    {:f,32} -> new_gpu_nx_nif(l,c,Kernel.to_charlist("float"))
    {:f,64} -> new_gpu_nx_nif(l,c,Kernel.to_charlist("double"))
    {:s,32} -> new_gpu_nx_nif(l,c,Kernel.to_charlist("int"))
    x -> raise "new_gmatrex: type #{x} not suported"
 end

 {:nx, type, {l,c}, [nil] , ref}
end

def get_gnx_({:nx, type, shape, name , ref}) do
  {l,c} = shape
  ref = case type do
    {:f,32} -> get_nx_nif(ref,l,c,Kernel.to_charlist("float"))
    {:f,64} -> get_nx_nif(ref,l,c,Kernel.to_charlist("double"))
    {:s,32} -> get_nx_nif(ref,l,c,Kernel.to_charlist("int"))
    x -> raise "new_gnx: type #{x} not suported"
 end

  %Nx.Tensor{data: %Nx.BinaryBackend{ state: ref}, type: type, shape: shape, names: name}
end

def new_gpu_nx_nif(_l,_c,_type) do
  raise "NIF get_nx_nif/4 not implemented"
end
def get_nx_nif(_matrex,_l,_c,_type) do
  raise "NIF get_nx_nif/4 not implemented"
end
def create_nx_ref_nif(_matrex,_l,_c,_type) do
  raise "NIF create_nx_ref_nif/4 not implemented"
end
  def create_ref_nif(_matrex) do
    raise "NIF create_ref_nif/1 not implemented"
end


################################
#######
#######  GMatrex stuff:
#######
####################

def new_pinned_nif(_list,_length) do
  raise "NIF new_pinned_nif/1 not implemented"
end
def new_gmatrex_pinned_nif(_array) do
  raise "NIF new_gmatrex_pinned_nif/1 not implemented"
end
def new_pinned(list) do
  size = length(list)
  {new_pinned_nif(list,size), {1,size}}
end
def new_gmatrex(%Matrex{data: matrix} = a) do
  ref=create_ref_nif(matrix)
  {ref, Matrex.size(a)}
end

def new_gmatrex((%Nx.Tensor{data: data, type: type, shape: shape, names: name}) ) do
  %Nx.BinaryBackend{ state: array} = data
  {l,c} = shape
  ref = case type do
     {:f,32} -> create_nx_ref_nif(array,l,c,Kernel.to_charlist("float"))
     {:f,64} -> create_nx_ref_nif(array,l,c,Kernel.to_charlist("double"))
     {:s,32} -> create_nx_ref_nif(array,l,c,Kernel.to_charlist("int"))
     x -> raise "new_gmatrex: type #{x} not suported"
  end
  {:nx, type, shape, name , ref}
end
def new_gmatrex({array,{l,c}}) do
  ref=new_gmatrex_pinned_nif(array)
  {ref, {l,c}}
end

def new_gmatrex(r,c) do
  ref=new_ref_nif(c)
  {ref, {r,c}}
  end

def gmatrex_size({_r,{l,size}}), do: {l,size}

def new_ref_nif(_matrex) do
  raise "NIF new_ref_nif/1 not implemented"
end
def synchronize_nif() do
  raise "NIF new_ref_nif/1 not implemented"
end
def synchronize() do
  synchronize_nif()
end
def new_ref(size) do
ref=new_ref_nif(size)
{ref, {1,size}}
end
def get_matrex_nif(_ref,_rows,_cols) do
raise "NIF get_matrex_nif/1 not implemented"
end
def get_gmatrex({ref,{rows,cols}}) do
  %Matrex{data: get_matrex_nif(ref,rows,cols)}
end
def get_gmatrex({:nx, type, shape, name , ref}) do
  {rows,cols} = shape
  %Nx.Tensor{data: %Nx.BinaryBackend{ state: get_matrex_nif(ref,rows,cols)}, type: type, shape: shape, names: name}
end

def load_kernel_nif(_module,_fun) do
  raise "NIF load_kernel_nif/2 not implemented"
end
def load_fun_nif(_module,_fun) do
  raise "NIF load_fun_nif/2 not implemented"
end

############################################################## Loading types and asts from files
def load_ast(kernel) do
  {_module,f_name}= case Macro.escape(kernel) do
    {:&, [],[{:/, [], [{{:., [], [module, f_name]}, [no_parens: true], []}, _nargs]}]} -> {module,f_name}
    f -> {:ok,f}
     #_ -> raise "Argument to spawn should be a function."
  end

 # bytes = File.read!("c_src/#{module}.asts")
 # map_asts = :erlang.binary_to_term(bytes)
 # IO.inspect map_size(map_asts)
 # {ast,_typed?,_types} = Map.get(map_asts,String.to_atom("#{f_name}"))
 # ast

 send(:module_server,{:get_ast,f_name,self()})

 receive do
   {:ast,ast} -> ast
    h     -> raise "unknown message for function type server #{inspect h}"
 end
end

def load_type_ast(kernel) do
  {:&, _ ,[{:/, _,  [{{:., _, [{:__aliases__, _, [module]}, kernelname]}, _, []}, _nargs]}]} = kernel
  bytes = File.read!("c_src/Elixir.#{module}.types")
  map_types = :erlang.binary_to_term(bytes)

              #module_name=String.slice("#{module}",7..-1//1) # Eliminates Elixir.
  type = Map.get(map_types,String.to_atom("#{kernelname}"))

  bytes = File.read!("c_src/Elixir.#{module}.asts")
  map_asts = :erlang.binary_to_term(bytes)

            #module_name=String.slice("#{module}",7..-1//1) # Eliminates Elixir.
  ast = Map.get(map_asts,String.to_atom("#{kernelname}"))
  {type,ast}
end

def load_type(kernel) do
  case Macro.escape(kernel) do
    {:&, [],[{:/, [], [{{:., [], [module, kernelname]}, [no_parens: true], []}, _nargs]}]} ->
             #IO.inspect module

              bytes = File.read!("c_src/#{module}.types")
              map = :erlang.binary_to_term(bytes)

              #module_name=String.slice("#{module}",7..-1//1) # Eliminates Elixir.


              resp = Map.get(map,String.to_atom("#{kernelname}"))
              #IO.inspect resp
              resp
    _ -> raise "PolyHok.build: invalid kernel"
  end
end
def load(kernel) do
  case Macro.escape(kernel) do
    {:&, [],[{:/, [], [{{:., [], [_module, kernelname]}, [no_parens: true], []}, _nargs]}]} ->


             # IO.puts module
              #raise "hell"
              #module_name=String.slice("#{module}",7..-1//1) # Eliminates Elixir.
              PolyHok.load_kernel_nif(to_charlist("Elixir.App"),to_charlist("#{kernelname}"))

    _ -> raise "PolyHok.build: invalid kernel"
  end
end
def load_fun(fun) do
  case Macro.escape(fun) do
    {:&, [],[{:/, [], [{{:., [], [_module, funname]}, [no_parens: true], []}, _nargs]}]} ->

              #module_name=String.slice("#{module}",7..-1//1) # Eliminates Elixir.

              PolyHok.load_fun_nif(to_charlist("Elixir.App"),to_charlist("#{funname}"))
    _ -> raise "PolyHok.invalid function"
  end
end
def load_lambda_compilation(_module,lambda,type) do
 # {:anon, lambda, PolyHok.load_fun_nif(to_charlist(module),to_charlist(lambda)), type}
 {:anon, lambda, type}
end
def load_lambda(lambda) do
  PolyHok.load_fun_nif(to_charlist("Elixir.App"),to_charlist(lambda))
 end
############################
######
######   Prepares the  arguments before making the real kernell call
######
##############
defp process_args([{:anon,name,_type}|t1]) do
  [load_lambda(name) | process_args(t1)]
end
defp process_args([{:func, func, _type}|t1]) do
  [load_fun(func)| process_args(t1)]
end
defp process_args([{:nx, _type, _shape, _name , ref}|t1]) do
  [ref| process_args(t1)]
end
defp process_args([{matrex,{_rows,_cols}}| t1]) do
  [matrex | process_args(t1)]
end
defp process_args([arg|t1]) when is_function(arg) do
  [load_fun(arg)| process_args(t1)]
end
defp process_args([arg|t1]) do
  [arg | process_args(t1)]
end
defp process_args([]), do: []

#########################
defp process_args_no_fun([{:anon,_name,_type}|t1]) do
  process_args_no_fun(t1)
end
defp process_args_no_fun([{:func, _func, _type}|t1]) do
  process_args_no_fun(t1)
end
defp process_args_no_fun([ {:nx, _type, _shape, _name , ref}| t1]) do
  [ref | process_args_no_fun(t1)]
end
defp process_args_no_fun([{matrex,{_rows,_cols}}| t1]) do
  [matrex | process_args_no_fun(t1)]
end
defp process_args_no_fun([arg|t1]) when is_function(arg) do
  process_args_no_fun(t1)
end
defp process_args_no_fun([arg|t1]) do
  [arg | process_args_no_fun(t1)]
end
defp process_args_no_fun([]), do: []

############################ Type checking the arguments at runtime

### first two arguments are used for error messages. Takes a list of types and a list of actual parameters and type checks them

def type_check_args(kernel,narg, [:matrex | t1], [a|t2]) do
    case a do
      {_ref,{_l,_c}} -> type_check_args(kernel,narg+1,t1,t2)
      {:nx, _type, _shape, _name , _ref} -> type_check_args(kernel,narg+1,t1,t2)
      _             -> raise "#{kernel}: argument #{narg} should have type gmatrex."
    end

end
def type_check_args(kernel,narg, [:float | t1], [v|t2]) do
    if is_float(v) do
      type_check_args(kernel,narg+1,t1,t2)
    else
      raise "#{kernel}: argument #{narg} should have type float."
    end
end
def type_check_args(kernel,narg, [:int | t1], [v|t2]) do
  if is_integer(v) do
    type_check_args(kernel,narg+1,t1,t2)
  else
    raise "#{kernel}: argument #{narg} should have type int."
  end
end
def type_check_args(kernel,narg, [{rt , ft} | t1], [{:func, func, { art , aft}} |t2]) do
  f_name= case Macro.escape(func) do
    {:&, [],[{:/, [], [{{:., [], [_module, f_name]}, [no_parens: true], []}, _nargs]}]} -> f_name
     _ -> raise "Argument to spawn should be a function."
   end
  if rt == art do

     type_check_function(f_name,0,ft,aft)
     type_check_args(kernel,narg+1,t1,t2)
   else
     raise "#{kernel}: #{f_name} function has return type #{art}, was excpected to have type #{rt}."
   end
  end

def type_check_args(kernel,narg, [{rt , ft} | t1], [{:anon, _name, { art , aft}} |t2]) do
  if rt == art do
    type_check_function("anonymous",1,ft,aft)
    type_check_args(kernel,narg+1,t1,t2)
  else
    raise "#{kernel}: anonymous function has return type #{art}, was excpected to have type #{rt}."
  end
end
def type_check_args(kernel,narg, [{rt , ft} | t1], [func |t2]) when is_function(func) do
  #IO.inspect func
  #raise "hell"
   {art,aft} = load_type(func)
   #IO.inspect ft
   #IO.inspect aft
   f_name= case Macro.escape(func) do
    {:&, [],[{:/, [], [{{:., [], [_module, f_name]}, [no_parens: true], []}, _nargs]}]} -> f_name
     _ -> raise "Argument to spawn should be a function."
   end
  if rt == art do
      type_check_function(f_name,0,ft,aft)
      type_check_args(kernel,narg+1,t1,t2)
    else
      raise "#{kernel}: #{f_name} function has return type #{art}, was excpected to have type #{rt}."
    end
end
def type_check_args(_k,_narg,[],[]), do: []
def type_check_args(k,_narg,a,v), do: raise "Wrong number of arguments when calling #{k}. #{inspect a} #{inspect v} "

def type_check_function(k,narg,[at|t1],[ft|t2]) do
    if (at == ft) do
      type_check_function(k,narg+1,t1,t2)
    else
      raise "#{k}: argument #{narg} has type #{ft} and should have type #{at}"
    end
end
def type_check_function(_k,_narg,[],[]), do: []
def type_check_function(k,_narg,a,v), do: raise "Wrong number of arguments when calling #{k}. #{inspect a} #{inspect v} "

####################### loads type of a function at compilation time
defmacro lt(k) do
  type = load_type_at_compilation(k)
  r= quote do: {:func, unquote(k), unquote(type)}
  #IO.inspect r
  r
end
def load_type_at_compilation(kernel) do
  {:&, _ ,[{:/, _,  [{{:., _, [{:__aliases__, _, [module]}, kernelname]}, _, []}, _nargs]}]} = kernel
#  {:&, [],[{:/, [], [{{:., [], [module, kernelname]}, [no_parens: true], []}, _nargs]}]} = kernel
  bytes = File.read!("c_src/Elixir.#{module}.types")
              map = :erlang.binary_to_term(bytes)

              #module_name=String.slice("#{module}",7..-1//1) # Eliminates Elixir.


              resp = Map.get(map,String.to_atom("#{kernelname}"))
              #IO.inspect resp
              resp
end
####################################


##################################
###########
###########  Spawn with jit compilation
###########
########################################

######## at compilation we build a representation for the kernel: {:ker, its type, its ast}
##### and leave a call to spawn

def spawn(k,t,b,l) do
  prev = System.monotonic_time()
  kernel_name = JIT.get_kernel_name(k)
  {kast,fun_graph} = case load_ast(k) do
            {a,g} -> {a,g}
            nil -> raise "Unknown kernel #{inspect kernel_name}"
  end
  delta = JIT.gen_types_delta(kast,l)
  #IO.inspect "Delta: #{inspect delta}"
  inf_types = JIT.infer_types(kast,delta)
  #IO.inspect "inf type: #{inspect inf_types}"
  #raise "hell"
  subs = JIT.get_function_parameters(kast,l)
  #IO.inspect subs
  kernel =JIT.compile_kernel(kast,inf_types,subs)
  #IO.inspect "kernel: #{inspect kernel}"
  funs=JIT.get_function_parameters_and_their_types(kast,l,inf_types)
  other_funs = fun_graph
                |> Enum.map(fn x -> {x, inf_types[x]} end)
                |> Enum.filter(fn {_,i} -> i != nil end)
  #IO.inspect funs
  #IO.inspect other_funs
  comp = Enum.map(funs++other_funs,&JIT.compile_function/1)
  comp = Enum.reduce(comp,[], fn x, y -> y++x end)
  #IO.puts comp
  includes = JIT.get_includes()
  prog = [includes| comp] ++[kernel]

  prog = Enum.reduce(prog,"", fn x, y -> y<>x end)

  #file = File.open!("c_src/Elixir.teste.cu", [:write])
  #IO.write(file, prog)

  #out = :os.cmd('echo \'#{prog}\' | nvcc -x cu --shared --compiler-options \'-fPIC\' -o priv/Elixir.#{kernel_name}.so -')

 # if out != [] do raise "Error when compiling .cu file generated by PolyHok:\n#{out}" end

  #IO.puts "COMPILATION: #{out}"

  args = process_args_no_fun(l)
  types_args = JIT.get_types_para(kast,inf_types)

  #(_n,_k,_t,_b,_size,_types,_l)

  #IO.puts prog
  next = System.monotonic_time()
  IO.puts "Poly\t#{kernel_name}\t#{System.convert_time_unit(next-prev,:native,:microsecond)/1000.0}"

  jit_compile_and_launch_nif(Kernel.to_charlist(kernel_name),Kernel.to_charlist(prog),t,b, length(args), types_args,args)
  #IO.inspect args

  #IO.puts out
  #{result, errcode} = System.cmd("sh",
  #    [ "-c echo \'#{inspect prog}\'","|","nvcc -x cu --shared --compiler-options '-fPIC' -o priv/Elixir.App.so -"
  #        #"c_src/Elixir.#{module_name}.cu"
  #], stderr_to_stdout: true)

  ##IO.inspect prog

  #IO.inspect errcode
  ##IO.inspect result
  #if ((errcode == 1) || (errcode ==2)) do raise "Error when compiling .cu file generated by PolyHok:\n#{result}" end




end



#############################################3
##########
########### spawn that uses function pointers
#########
#######################################

def spawn_st({:func, k, type}, t,b,l) do
  #IO.puts "Aqui!"
  f_name= case Macro.escape(k) do
    {:&, [],[{:/, [], [{{:., [], [_module, f_name]}, [no_parens: true], []}, _nargs]}]} -> f_name
     _ -> raise "Argument to spawn should be a function."
  end



    {:unit,tk} = type

    type_check_args(f_name,1,tk,l)



    args = process_args(l)
    pk=load(k)
    spawn_nif(pk,t,b,args)

end

def spawn_st(k,t,b,l) when is_function(k) do
   #IO.inspect k
   #raise "hell"

  f_name= case Macro.escape(k) do
    {:&, [],[{:/, [], [{{:., [], [_module, f_name]}, [no_parens: true], []}, _nargs]}]} -> f_name
     _ -> raise "Argument to spawn should be a function."
  end



    {:unit,tk} = load_type(k)

    type_check_args(f_name,1,tk,l)

    pk=load(k)

    args = process_args(l)

    spawn_nif(pk,t,b,args)

end
def spawn_st(_k,_t,_b,_l) do
  #IO.inspect _k
  raise "First argument of spawn must be a function.."
end
def spawn_nif(_k,_t,_b,_l) do
  raise "NIF spawn_nif/1 not implemented"
end

def jit_compile_and_launch_nif(_n,_k,_t,_b,_size,_types,_l) do
  raise "NIF jit_compile_and_launch_nif/7 not implemented"
end
end
