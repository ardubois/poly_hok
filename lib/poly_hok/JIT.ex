require PolyHok.CudaBackend

defmodule JIT do

  def compile_function({:anon,fname,code,type}) do
   {code,fun_graph} = code
   
    #IO.puts "Compile function: #{fname}"

    delta = gen_delta_from_type(code,type)
   # IO.inspect "Delta: #{inspect delta}"

    inf_types = JIT.infer_types(code,delta)
    #IO.inspect "inf_types function: #{inspect inf_types}"
    {:fn, _, [{:->, _ , [para,body]}] } = code


    param_list = para
        |> Enum.map(fn {p, _, _}-> PolyHok.CudaBackend.gen_para(p,Map.get(inf_types,p)) end)
        |> Enum.join(", ")

    param_vars = para
        |>  Enum.map(fn {p, _, _}-> p end)


    fun_type =  Map.get(inf_types,:return)
    fun_type = if (fun_type == :unit) do :void else fun_type end

    cuda_body = PolyHok.CudaBackend.gen_cuda_jit(body,inf_types,param_vars,"module",MapSet.new())
    k =        PolyHok.CudaBackend.gen_function(fname,param_list,cuda_body,fun_type)

    function = "\n" <> k <> "\n\n"

    other_funs = fun_graph
                |> Enum.map(fn x -> {x, inf_types[x]} end)
                |> Enum.filter(fn {_,i} -> i != nil end)
    comp = Enum.map(other_funs,&JIT.compile_function/1)
    comp = Enum.reduce(comp,[], fn x, y -> y++x end)
    comp ++ [function]
end
def compile_function({name,type}) do
  #IO.puts "Compile function: #{name}"
  nast = PolyHok.load_ast(name)
  case nast do
    nil -> [""]
    {fast,fun_graph} ->
          delta = gen_delta_from_type(fast,type)
         #IO.inspect "Delta: #{inspect delta}"
         # IO.inspect "Type: #{inspect type}"
    #      IO.inspect "Call graph: #{inspect fun_graph}"
          inf_types = JIT.infer_types(fast,delta)
         #IO.inspect "inf_types function: #{inspect inf_types}"
          {:defd,_iinfo,[header,[body]]} = fast
        #  IO.inspect body
          {fname, _, para} = header

          param_list = para
              |> Enum.map(fn {p, _, _}-> PolyHok.CudaBackend.gen_para(p,Map.get(inf_types,p)) end)
              |> Enum.join(", ")

          param_vars = para
              |>  Enum.map(fn {p, _, _}-> p end)


          fun_type =  Map.get(inf_types,:return)
          fun_type = if (fun_type == :unit) do :void else fun_type end

          cuda_body = PolyHok.CudaBackend.gen_cuda_jit(body,inf_types,param_vars,"module",MapSet.new())
          k =        PolyHok.CudaBackend.gen_function(fname,param_list,cuda_body,fun_type)

          function = "\n" <> k <> "\n\n"

          other_funs = fun_graph
                |> Enum.map(fn x -> {x, inf_types[x]} end)
                |> Enum.filter(fn {_,i} -> i != nil end)
          #IO.inspect funs
          #IO.inspect "inf_types: #{inspect inf_types}"
          #IO.inspect "other funs: #{inspect other_funs}"
          comp = Enum.map(other_funs,&JIT.compile_function/1)
          #IO.inspect "Comp: #{inspect comp} "
          comp = Enum.reduce(comp,[], fn x, y -> y++x end)
          comp ++ [function]
    end
end

def compile_kernel({:defk,_,[header,[body]]},inf_types,subs) do

  {fname, _, para} = header

  param_list = para
      |> Enum.map(fn {p, _, _}-> PolyHok.CudaBackend.gen_para(p,Map.get(inf_types,p)) end)
      |> Enum.filter(fn p -> p != nil end)
      |> Enum.join(", ")


  param_vars = para
   |>  Enum.map(fn {p, _, _}-> p end)

  #types_para = para
  # |>  Enum.map(fn {p, _, _}-> Map.get(inf_types,p) end)
  # |> Enum.filter(fn p -> case p do
  #                           {_,_} -> false
  #                            _ -> true
  #                  end end)

   cuda_body = PolyHok.CudaBackend.gen_cuda_jit(body,inf_types,param_vars,"module",subs)
   k = PolyHok.CudaBackend.gen_kernel_jit(fname,param_list,cuda_body)
   #accessfunc = PolyHok.CudaBackend.gen_kernel_call(fname,length(types_para),Enum.reverse(types_para))
   #IO.puts accessfunc
   "\n" <> k <> "\n\n" # <> accessfunc
end

def gen_delta_from_type( {:defd,_,[header,[_body]]}, {return_type, types} ) do
   {_, _, formal_para} = header
   delta=formal_para
          |> Enum.map(fn({p, _, _}) -> p end)
          |> Enum.zip(types)
          |> Map.new()
   Map.put(delta, :return, return_type)
end
def gen_delta_from_type( {:fn, _, [{:->, _ , [para,_body]}] }, {return_type, types} ) do

  delta=para
         |> Enum.map(fn({p, _, _}) -> p end)
         |> Enum.zip(types)
         |> Map.new()
  Map.put(delta, :return, return_type)
end
def get_types_para({:defk,_,[header,[_body]]},  delta) do
  {_, _, formal_para} = header
  formal_para
          |> Enum.map(fn({p, _, _}) -> delta[p] end)
          |> Enum.filter(fn p -> case p do
                                  {_, _} -> false
                                      _ -> true
                                  end
                                  end)
          |> Enum.map(fn x -> Kernel.to_charlist(to_string(x)) end)
end
def get_function_parameters_and_their_types({:defk,_,[header,[_body]]}, actual_para, delta) do
  {_, _, formal_para} = header
  formal_para
          |> Enum.map(fn({p, _, _}) -> p end)
          |> Enum.zip(actual_para)
          |> Enum.filter(fn {_n,p} -> is_function_para(p) end)
          |> Enum.map(fn {n,p} -> case p do
                                      {:anon, name, code} -> {:anon, name, code, delta[n]}
                                      _ -> {get_function_name(p),delta[n]}
                                  end
                                  end)
end
def get_function_parameters({:defk,_,[header,[_body]]}, actual_para) do
  {_, _, formal_para} = header
  formal_para
          |> Enum.map(fn({p, _, _}) -> p end)
          |> Enum.zip(actual_para)
          |> Enum.filter(fn {_n,p} -> is_function_para(p) end)
          |> Enum.reduce( Map.new(), fn {n,p}, map -> Map.put(map,n,get_function_name(p)) end)
         # |> Enum.map(fn {n,p} -> {n,p} end)
end
def is_anon(func) do
  case func do
    {:anon,_name,_code} -> true
    _ -> false
  end
end
def is_function_para(func) do
  case func do
    {:anon, _name, _code} -> true
    func when is_function(func) -> true
    _h -> false
  end
end
def get_function_name({:anon, name, _code}) do
  name
end
def get_function_name(fun) do
  {_module,f_name}= case Macro.escape(fun) do
    {:&, [],[{:/, [], [{{:., [], [module, f_name]}, [no_parens: true], []}, _nargs]}]} -> {module,f_name}
     _ -> raise "Argument to spawn should be a function: #{inspect Macro.escape(fun)}"
  end
  f_name
end
def infer_types({:defk,_,[_header,[body]]},delta) do
  PolyHok.TypeInference.type_check(delta,body)
end
def infer_types({:defd,_,[_header,[body]]},delta) do
  PolyHok.TypeInference.type_check(delta,body)
end
def infer_types({:fn, _, [{:->, _ , [_para,body]}] },delta) do

  PolyHok.TypeInference.type_check(delta,body)
end
  # finds the types of the actual parameters and generates a maping of formal parameters to their types
def gen_types_delta({:defk,_,[header,[_body]]}, actual_param) do
  {_, _, formal_para} = header
  types = infer_types_actual_parameters(actual_param)
  formal_para
          |> Enum.map(fn({p, _, _}) -> p end)
          |> Enum.zip(types)
          |> Map.new()
end


def infer_types_actual_parameters([])do
  []
end
def infer_types_actual_parameters([h|t])do
  case h do
    {:nx, type, _shape, _name , _ref} ->
        case type do
          {:f,32} -> [:tfloat | infer_types_actual_parameters(t)]
          {:f,64} -> [:tdouble | infer_types_actual_parameters(t)]
          {:s,32} -> [:tint | infer_types_actual_parameters(t)]
        end
    {:matrex, _kref, _size} -> [:tfloat | infer_types_actual_parameters(t)]
    {:anon,_name,_code} ->     [:none | infer_types_actual_parameters(t)]
    float when  is_float(float) ->
        [:float | infer_types_actual_parameters(t)]
    int   when  is_integer(int) ->
        [:int | infer_types_actual_parameters(t)]
    func when is_function(func) ->
        [:none | infer_types_actual_parameters(t)]
  end
end

def get_includes() do
  send(:module_server,{:get_include,self()})

 inc=receive do
   {:include,inc} -> inc
    h     -> raise "unknown message for function type server #{inspect h}"
 end
 case inc do
  nil -> ""
  list -> Enum.reduce(list,"", fn x, y -> y<>x end)
 end
end
def load_kernel(kernel) do
  case Macro.escape(kernel) do
    {:&, [],[{:/, [], [{{:., [], [_module, kernelname]}, [no_parens: true], []}, _nargs]}]} ->


             # IO.puts module
              #raise "hell"
              #module_name=String.slice("#{module}",7..-1//1) # Eliminates Elixir.
              PolyHok.load_kernel_nif(to_charlist("Elixir.#{kernelname}"),to_charlist("#{kernelname}"))

    _ -> raise "PolyHok.build: invalid kernel"
  end
end
def get_kernel_name(kernel) do
  case Macro.escape(kernel) do
    {:&, [],[{:/, [], [{{:., [], [_module, kernelname]}, [no_parens: true], []}, _nargs]}]} -> kernelname



    _ -> raise "PolyHok.build: invalid kernel"
  end
end
#####
### Processes a module and populates the ast server with information about functions (their ast, and call graph)
#################

def process_module(module_name,body) do

  # initiate server that collects types and asts
  if (Process.whereis(:module_server) == nil) do
    pid = spawn_link(fn -> module_server(%{}, %{}, %{}) end)
    try do
         Process.register(pid, :module_server)
    rescue
    _ -> :ok  
     end
  end   
  _defs=case body do
      {:__block__, [], definitions} ->  process_definitions(module_name,definitions,[])
      _   -> process_definitions(module_name,[body],[])
  end
end

###########################
######  This server constructs two maps: 1. function names ->  types
#                                        2. function names -> ASTs
######            Types are used to type check at runtime a kernel call
######            ASTs are used to recompile a kernel at runtime substituting the names of the formal parameters of a function for
######         the actual parameters
############################

def module_server(types_map, ast_map, ptx_map) do
  receive do
    {:add_ast, fun, ast, funs} ->
      module_server(types_map, Map.put(ast_map,fun,{ast,funs}), ptx_map)
    {:get_ast, f_name, pid} ->
      send(pid, {:ast, ast_map[f_name]})
      module_server(types_map, ast_map, ptx_map)
    {:add_type, fun, type} ->
      module_server(Map.put(types_map, fun, type), ast_map, ptx_map)
    {:get_map, pid} ->
    send(pid, {:map, {types_map,ast_map}})
      module_server(types_map, ast_map, ptx_map)
    {:get_include, pid} ->
    send(pid, {:include, ast_map[:include]})
      module_server(types_map, ast_map, ptx_map)
    {:add_include, inc} ->
      case ast_map[:include] do
        nil -> module_server(types_map, Map.put(ast_map, :include, [inc]), ptx_map)
        l -> module_server(types_map, Map.put(ast_map, :include, [inc | l]), ptx_map)
      end
    {:get_ptx, kernel_name, pid} ->
      send(pid, {:ptx, ptx_map[kernel_name]})
      module_server(types_map, ast_map, ptx_map)
    {:set_ptx, kernel_name, ptx} ->
      module_server(types_map, ast_map, Map.put(ptx_map, kernel_name, ptx))
    {:kill} -> :ok
  end
end

#############################################
##### For every function and kernel definition, it registers an ast and and the functions called inside the definition
#####################
defp process_definitions(_module_name, [],_l), do: :ok
defp process_definitions(module_name,[h|t],l) do
       case h do
        {:defk,_,[header,[_body]]} ->  {fname, _, _para} = header
                                      funs = find_functions(h)
                                      register_function(module_name,fname,h,funs)
                                      process_definitions(module_name,t,[{module_name,fname,h,funs}|t])

        {:defd , ii, [header,[body]]} -> {fname, _, _para} = header

                                      #  IO.inspect "Process definitions: #{fname}"


                                        body = PolyHok.TypeInference.add_return(Map.put(%{}, :return, :none), body)
                                        #body = PolyHok.CudaBackend.add_return( body)
                                   #  IO.inspect body
                                        funs = find_functions({:defd , ii, [header,[body]]})
                                       # IO.inspect "Function graph: #{inspect funs}"
                                       # IO.inspect "body: #{inspect body}"
                                        register_function(module_name,fname,{:defd , ii, [header,[body]]},funs)
                                        process_definitions(module_name,t, [{module_name,fname,{:defd , ii, [header,[body]]},funs}|l])
        {:include, _, [{_,_,[name]}]} ->
                code = File.read!("c_src/Elixir.#{name}.cu")
                #IO.inspect code
                send(:module_server,{:add_include,code})
                process_definitions(module_name,t,l)
        _               -> process_definitions(module_name,t,l)


      end

end

def register_function(_module_name,fun_name,ast,funs) do
  send(:module_server,{:add_ast,fun_name,ast,funs})
end

###################
#### finds the names of functions called inside a device function or kernel
########################

def find_functions({:fn, _aa, [{:->, _bb , [para,body]}]}) do
  param_vars = para
  |>  Enum.map(fn {p, _, _}-> p end)
  |>  MapSet.new()

  {_args,funs} = find_function_calls_body({param_vars,MapSet.new()},body)

  MapSet.to_list(funs)
end

def find_functions({:defk, _i1,[header, [body]]}) do
  {_fname, _, para} = header

  param_vars = para
  |>  Enum.map(fn {p, _, _}-> p end)
  |>  MapSet.new()

  {_args,funs} = find_function_calls_body({param_vars,MapSet.new()},body)

  MapSet.to_list(funs)
end


def find_functions({:defd, _i1,[header, [body]]}) do
 # IO.inspect "aqui inicio"
  {_fname, _, para} = header

  param_vars = para
  |>  Enum.map(fn {p, _, _}-> p end)
  |>  MapSet.new()

  #IO.inspect "body #{inspect body}"
  {_args,funs} = find_function_calls_body({param_vars,MapSet.new()},body)

  MapSet.to_list(funs)
end

def find_function_calls_body(map,body) do

  case body do
     {:__block__, _, _code} ->
      find_function_calls_block(map,body)
     {:do, {:__block__,pos, code}} ->
      find_function_calls_block(map, {:__block__, pos,code})
     {:do, exp} ->
     # IO.inspect "here #{inspect exp}"
      find_function_calls_command(map,exp)
     {_,_,_} ->
      find_function_calls_command(map,body)
  end
end


defp find_function_calls_block(map,{:__block__, _info, code}) do
  Enum.reduce(code,map, fn x,acc -> find_function_calls_command(acc,x) end)
end

defp find_function_calls_command(map,code) do
  #IO.inspect "here2"
  case code do
      {:for,_i,[_param,[body]]} ->
       find_function_calls_body(map,body)
      {:do_while, _i, [[doblock]]} ->
       find_function_calls_body(map,doblock)
      {:do_while_test, _i, [exp]} ->
       find_function_calls_exp(map,exp)
      {:while, _i, [bexp,[body]]} ->
       map = find_function_calls_exp(map,bexp)
       find_function_calls_body(map,body)
      # CRIAÇÃO DE NOVOS VETORES
      {{:., _i1, [Access, :get]}, _i2, [arg1,arg2]} ->
        map=find_function_calls_exp(map,arg1)
        find_function_calls_exp(map,arg2)
      {:__shared__, _i1, [{{:., _i2, [Access, :get]}, _i3, [arg1,arg2]}]} ->
        map=find_function_calls_exp(map,arg1)
        find_function_calls_exp(map,arg2)

      # assignment
      {:=, _i1, [{{:., _i2, [Access, :get]}, _i3, [{_array,_a1,_a2},acc_exp]}, exp]} ->
        map= find_function_calls_exp(map,acc_exp)
        find_function_calls_exp(map,exp)
      {:=, _i, [_var, exp]} ->
       find_function_calls_exp(map,exp)
      {:if, _i, if_com} ->
       find_function_calls_if(map,if_com)
      {:var, _i1 , [{_var,_i2,[{:=, _i3, [{_type,_ii,nil}, exp]}]}]} ->
       find_function_calls_exp(map,exp)
      {:var, _i1 , [{_var,_i2,[{:=, _i3, [_type, exp]}]}]} ->
       find_function_calls_exp(map,exp)
      {:var, _i1 , [{_var,_i2,[{_type,_i3,_t}]}]} ->
        map
      {:var, _i1 , [{_var,_i2,[_type]}]} ->
        map
      {:type, _i1 , [{_var,_i2,[{_type,_i3,_t}]}]} ->
        map
      {:type, _i1 , [{_var,_i2,[_type]}]} ->
        map

      {:return,_i,[arg]} ->
   #     IO.inspect "Aqui3"
       find_function_calls_exp(map,arg)

      {fun, _info, args} when is_list(args)->
    #    IO.inspect "Aqui3 #{length args} #{inspect fun}"
        {args,funs} = map
        if MapSet.member?(args,fun) do
          map
        else
           {args,MapSet.put(funs,fun)}
        end
      number when is_integer(number) or is_float(number) -> raise "Error: #{inspect number} is a command"
      {str,i1 ,a } -> {str,i1 ,a }

  end
end

defp find_function_calls_if(map,[bexp, [do: then]]) do
  map=find_function_calls_exp(map,bexp)
  find_function_calls_body(map,then)
 end
 defp find_function_calls_if(map,[bexp, [do: thenbranch, else: elsebranch]]) do
  map=find_function_calls_exp(map,bexp)
  map=find_function_calls_body(map,thenbranch)
  find_function_calls_body(map,elsebranch)
 end


 defp find_function_calls_exp(map,exp) do
  case exp do
    {{:., _i1, [Access, :get]}, _i2, [_arg1,arg2]} ->
     find_function_calls_exp(map,arg2)
    {{:., _i1, [{_struct, _i2, nil}, _field]},_i3,[]} ->
        map
    {{:., _i1, [{:__aliases__, _i2, [_struct]}, _field]}, _i3, []} ->
       map
    {op,_info, args} when op in [:+, :-, :/, :*] ->
     # IO.inspect "Aqui"
      Enum.reduce(args,map, fn x,acc -> find_function_calls_exp(acc,x) end)

    {op, _info, args} when op in [ :<=, :<, :>, :>=, :&&, :||, :!,:!=,:==] ->
      Enum.reduce(args,map, fn x,acc -> find_function_calls_exp(acc,x) end)
    {var,_info, nil} when is_atom(var) ->
       map
    {fun,_info, _args} ->
      #IO.inspect "Aqui2"
     {args,funs} = map
     if MapSet.member?(args,fun) do
       map
     else
        {args,MapSet.put(funs,fun)}
     end
    float when  is_float(float) -> map
    int   when  is_integer(int) -> map
    string when is_binary(string)  -> map
  end

end
#############################################################
###############################
##########
############  BEGIN CLOSURE ELIMINATION
############
################


def closure_elimination(kast,l) do
  ###### process closure
  ##### adds extra parameters to kernels
  ####  adds extra arguments to closure calls
  ###   adds extra arguments to the kernel call
  #############
  if (contains_closure(l))do
  # IO.puts "yooooooooooooooooooooooooo"
   map_extra_args =  JIT.gen_map_fun_name_to_extra_para(kast,l)
   #IO.inspect map_extra_args
   kast = JIT.add_extra_closure_args(map_extra_args,kast)
   #IO.inspect kast
   #raise "hrell"
   kast = JIT.add_extra_closure_param(kast,l)
   l = JIT.add_extra_args_from_closures(l)
   {kast,l}
  else
    {kast,l}
  end
end

def contains_closure([]), do: false
def contains_closure([{:closure,_name,_ast,_cargs,_cvalues}|_lt]), do: true
def contains_closure([_a|lt]), do: contains_closure(lt)

#########  creates a map from function arguments to the kernel to their extra parameters

def gen_map_fun_name_to_extra_para({:defk,_info,[header,[_body]]},l) do
  {_fname, _info_header, para} = header
  param_vars = para
        |>  Enum.map(fn {p, _, _}-> p end)

  Map.new(gen_map_to_extra_para(param_vars,l))
end

defp gen_map_to_extra_para([],[]) do
  []
end
defp gen_map_to_extra_para([para|tpara],[{:closure,_name,_ast,cargs,_cvalues}|targ]) do
 # extra_args = Enum.map(cargs,fn p -> {p,[],nil} end)

  [{para, cargs}| gen_map_to_extra_para(tpara,targ)]
end
defp gen_map_to_extra_para([_para|tpara],[_arg|targ]) do
  gen_map_to_extra_para(tpara,targ)
end
defp gen_map_to_extra_para(_p,_a) do
  raise "Kernel is receiving a list or arguments with the wrong size!"
end
#### adds extra closure parameters to kernel parameters

def add_extra_closure_param({:defk,info,[header,[body]]},l) do
  {fname, info_header, para} = header
  new_param_list = gen_extra_closure_param(para,l)
  {:defk,info,[{fname, info_header, new_param_list},[body]]}
end
defp gen_extra_closure_param([],[]) do
  []
end
defp gen_extra_closure_param([para|tpara],[{:closure,_name,_ast,cargs,_cvalues}|targ]) do
  extra_args = Enum.map(cargs,fn p -> {p,[],nil} end)

  [para| extra_args] ++ gen_extra_closure_param(tpara,targ)
end
defp gen_extra_closure_param([para|tpara],[_arg|targ]) do
  [para |gen_extra_closure_param(tpara,targ)]
end
defp gen_extra_closure_param(_p,_a) do
  raise "Kernel is receiving a list or arguments with the wrong size!"
end

####### adds extra arguments from closures to the list of arguments of
####### kernel and substitute closures by anonymous functions

def add_extra_args_from_closures([]), do: []
def add_extra_args_from_closures([{:closure,name,ast,_cargs,cvalues}|targ]) do
  [{:anon,name,ast}|cvalues] ++ add_extra_args_from_closures(targ)
end
def add_extra_args_from_closures([arg|targ]) do
  [arg| add_extra_args_from_closures(targ)]
end




##################### FIND FREE VARIABLES ##############################
#################################################################


def find_free_vars({:fn, _, [{:->, _ , [para,body]}] }) do
  #IO.puts "para"
  #IO.inspect para
  #raise "hell"
  param_vars = para
    |>  Enum.map(fn {p, _, _}-> p end)
    |>  MapSet.new()

  #IO.inspect "body #{inspect body}"
    {_bound,free} = find_free_vars_body({param_vars,MapSet.new()},body)
   # IO.puts "Bound"
   # IO.inspect bound
   # IO.puts "Free"
   # IO.inspect free
    MapSet.to_list(free)
end



def find_free_vars_body(map,body) do

  case body do
     {:__block__, _, _code} ->
      find_free_vars_block(map,body)
     {:do, {:__block__,pos, code}} ->
      find_free_vars_block(map, {:__block__, pos,code})
     {:do, exp} ->
     # IO.inspect "here #{inspect exp}"
      find_free_vars_command(map,exp)
     {_,_,_} ->
      find_free_vars_command(map,body)
  end
end


defp find_free_vars_block(map,{:__block__, _info, code}) do
  Enum.reduce(code,map, fn x,acc -> find_free_vars_command(acc,x) end)
end


defp find_free_vars_command(map,code) do
  #IO.inspect "here2"
  case code do
      {:for,i,[param,[body]]} ->
       find_free_vars_for(map,{:for,i,[param,[body]]})
      {:do_while, _i, [[doblock]]} ->
       find_free_vars_body(map,doblock)
      {:do_while_test, _i, [exp]} ->
       find_free_vars_exp(map,exp)
      {:while, _i, [bexp,[body]]} ->
       map = find_free_vars_exp(map,bexp)
       find_free_vars_body(map,body)
      # CRIAÇÃO DE NOVOS VETORES
      {{:., _i1, [Access, :get]}, _i2, [{var,_,nil},size]} ->
        {bound,free} = map
        bound = MapSet.put(bound,var)
        find_free_vars_exp({bound,free},size)
      {:__shared__, _i1, [{{:., _i2, [Access, :get]}, _i3, [{var,_,nil},size]}]} ->
        {bound,free} = map
        bound = MapSet.put(bound,var)
        find_free_vars_exp({bound,free},size)
      # assignment
      {:=, _i1, [{{:., _i2, [Access, :get]}, _i3, [{array,_i4,_arag}, acc_exp]}, exp]} ->
        {bound,free} = map
        if MapSet.member?(bound,array) do
          map
           |> find_free_vars_exp(acc_exp)
           |> find_free_vars_exp(exp)
        else
          {bound, MapSet.put(free,array)}
          |> find_free_vars_exp(acc_exp)
          |> find_free_vars_exp(exp)
        end
      {:=, _i, [{var,_,nil}, exp]} ->
        {bound,free} = map
        {MapSet.put(bound,var), free}
        |> find_free_vars_exp(exp)

      {:if, _i, if_com} ->
       find_free_vars_if(map,if_com)
      {:var, _i1 , [{var,_i2,[{:=, _i3, [{_type,_ii,nil}, exp]}]}]} ->
        {bound,free} = map
        {MapSet.put(bound,var),free}
        |> find_free_vars_exp(exp)
      {:var, _i1 , [{var,_i2,[{:=, _i3, [_type, exp]}]}]} ->
        {bound,free} = map
        {MapSet.put(bound,var),free}
        |> find_free_vars_exp(exp)
      {:var, _i1 , [{var,_i2,[{_type,_i3,_t}]}]} ->
        {bound,free} = map
        {MapSet.put(bound,var),free}

      {:var, _i1 , [{var,_i2,[_type]}]} ->
        {bound,free} = map
        {MapSet.put(bound,var),free}
      {:type, _i1 , [{_var,_i2,[{_type,_i3,_t}]}]} ->
        map
      {:type, _i1 , [{_var,_i2,[_type]}]} ->
        map

      {:return,_i,[arg]} ->
   #     IO.inspect "Aqui3"
       find_free_vars_exp(map,arg)

      {_fun, _info, args} when is_list(args)->
    #    IO.inspect "Aqui3 #{length args} #{inspect fun}"
         Enum.reduce(args,map, fn x,acc -> find_free_vars_exp(acc,x) end)
    #raise "funcao: #{inspect fun}"
     #   {args,funs} = map
     #   if MapSet.member?(args,fun) do
     #     map
     #   else
     #      {args,MapSet.put(funs,fun)}
     #   end
      number when is_integer(number) or is_float(number) -> raise "Error: #{inspect number} is a command"
      c -> raise "Found unknown command (#{inspect c}) while searching for free vars."

  end
end

defp find_free_vars_for({bound,free},{:for,_,[header,[body]]}) do
  case header do
    {:in, _,[{var,_,nil},{:range,_,[n]}]} ->
       {bound,free} = {MapSet.put(bound,var),free}
            |> find_free_vars_exp(n)
            |> find_free_vars_body(body)

        {MapSet.delete(bound,var),free}
    {:in, _,[{var,_,nil},{:range,_,[arg1,arg2]}]} ->
        {bound,free} =  {MapSet.put(bound,var),free}
            |> find_free_vars_exp(arg1)
            |> find_free_vars_exp(arg2)
            |> find_free_vars_body(body)

        {MapSet.delete(bound,var),free}
    {:in, _,[{var,_,nil},{:range,_,[arg1,arg2,step]}]} ->
       {bound,free} = {MapSet.put(bound,var),free}
            |> find_free_vars_exp(arg1)
            |> find_free_vars_exp(arg2)
            |> find_free_vars_exp(step)
            |> find_free_vars_body(body)

    {MapSet.delete(bound,var),free}
  end
end


defp find_free_vars_if(map,[bexp, [do: then]]) do
  map
  |> find_free_vars_exp(bexp)
  |> find_free_vars_body(then)
 end
 defp find_free_vars_if(map,[bexp, [do: thenbranch, else: elsebranch]]) do
  map
  |> find_free_vars_exp(bexp)
  |> find_free_vars_body(thenbranch)
  |> find_free_vars_body(elsebranch)
 end


 defp find_free_vars_exp(map,exp) do
  case exp do
    ## array index
    {{:., _i1, [Access, :get]}, _i2, [{array,_ia,_},index]} ->
      {bound,free} = map
      if MapSet.member?(bound,array) do
        map
         |> find_free_vars_exp(index)
      else
        {bound, MapSet.put(free,array)}
        |> find_free_vars_exp(index)
      end
      # cuda constant
    {{:., _i1, [{_struct, _i2, nil}, _field]},_i3,[]} ->
        map
        # cuda constant
    {{:., _i1, [{:__aliases__, _i2, [_struct]}, _field]}, _i3, []} ->
       map
    {op,_info, args} when op in [:+, :-, :/, :*] ->
     # IO.inspect "Aqui"
      Enum.reduce(args,map, fn x,acc -> find_free_vars_exp(acc,x) end)

    {op, _info, args} when op in [ :<=, :<, :>, :>=, :&&, :||, :!,:!=,:==] ->
      Enum.reduce(args,map, fn x,acc -> find_free_vars_exp(acc,x) end)
    {var,_info, nil} when is_atom(var) ->
      {bound,free} = map
      if MapSet.member?(bound,var) do
        map
      else
        {bound, MapSet.put(free,var)}
      end
    {_fun,_info, args} ->
      Enum.reduce(args,map, fn x,acc -> find_free_vars_exp(acc,x) end)
      #IO.inspect "Aqui2"
     # IO.inspect args
     # raise "function: #{inspect fun}"
    # {args,funs} = map
     #if MapSet.member?(args,fun) do
     #  map
     #else
      #  {args,MapSet.put(funs,fun)}
    # end
    float when  is_float(float) -> map
    int   when  is_integer(int) -> map
    string when is_binary(string)  -> map
  end

end

####################################################
############### adds extra parameteres when calling a closure
#### Takes a map (closure name -> extra para) and the ast,  and returns a new ast
########################
def add_extra_closure_args(map,{:defk, i1,[header, [body]]}) do
  nbody = add_extra_closure_args__body(map,body)
  {:defk, i1,[header, [nbody]]}
end


def add_extra_closure_args__body(map,body) do


 case body do

     {:__block__, _, _code} ->
       add_extra_closure_args__block(map,body)
     {:do, {:__block__,pos, code}} ->
       {:do, add_extra_closure_args__block(map, {:__block__, pos,code}) }
     {:do, exp} ->
       {:do, add_extra_closure_args__command(map,exp)}
     {_,_,_} ->
       add_extra_closure_args__command(map,body)
  end


end
defp add_extra_closure_args__block(map,{:__block__, info, code}) do
 {:__block__, info,
     Enum.map(code,  fn com -> add_extra_closure_args__command(map,com) end)
 }
end

defp add_extra_closure_args__command(map,code) do
   case code do
       {:for,i,[param,[body]]} ->
         {:for,i,[param,[add_extra_closure_args__body(map,body)]]}
       {:do_while, i, [[doblock]]} ->
         {:do_while, i, [[add_extra_closure_args__body(map,doblock)]]}
       {:do_while_test, i, [exp]} ->
         {:do_while_test, i, [add_extra_closure_args__exp(map,exp)]}
       {:while, i, [bexp,[body]]} ->
         {:while, i, [add_extra_closure_args__exp(map,bexp),[add_extra_closure_args__body(map,body)]]}
       # CRIAÇÃO DE NOVOS VETORES
       {{:., i1, [Access, :get]}, i2, [arg1,arg2]} ->
         {{:., i1, [Access, :get]}, i2, [add_extra_closure_args__exp(map,arg1),add_extra_closure_args__exp(map,arg2)]}
       {:__shared__, i1, [{{:., i2, [Access, :get]}, i3, [arg1,arg2]}]} ->
         {:__shared__,i1 , [{{:., i2, [Access, :get]}, i3, [add_extra_closure_args__exp(map,arg1),add_extra_closure_args__exp(map,arg2)]}]}

       # assignment
       {:=, i1, [{{:., i2, [Access, :get]}, i3, [{array,a1,a2},acc_exp]}, exp]} ->
         {:=, i1, [{{:., i2, [Access, :get]}, i3, [{array,a1,a2},add_extra_closure_args__exp(map,acc_exp)]}, add_extra_closure_args__exp(map,exp)]}
       {:=, i, [var, exp]} ->
         {:=, i, [var, add_extra_closure_args__exp(map,exp)]}
       {:if, i, if_com} ->
         {:if, i, add_extra_closure_args__if(map,if_com)}
       {:var, i1 , [{var,i2,[{:=, i3, [{type,ii,nil}, exp]}]}]} ->
         {:var, i1 , [{var,i2,[{:=, i3, [{type,ii,nil}, add_extra_closure_args__exp(map,exp)]}]}]}
       {:var, i1 , [{var,i2,[{:=, i3, [type, exp]}]}]} ->
         {:var, i1 , [{var,i2,[{:=, i3, [type, add_extra_closure_args__exp(map,exp)]}]}]}
       {:var, i1 , [{var,i2,[{type,i3,t}]}]} ->
         {:var, i1 , [{var,i2,[{type,i3,t}]}]}
       {:var, i1 , [{var,i2,[type]}]} ->
         {:var, i1 , [{var,i2,[type]}]}
       {:type, i1 , [{var,i2,[{type,i3,t}]}]} ->
         {:type, i1 , [{var,i2,[{type,i3,t}]}]}
       {:type, i1 , [{var,i2,[type]}]} ->
         {:type, i1 , [{var,i2,[type]}]}

       {:return,i,[arg]} ->
         {:return,i,[add_extra_closure_args__exp(map,arg)]}

       {fun, info, args} when is_list(args)->
         case map[fun] do
            nil -> {fun, info, Enum.map(args,fn(exp) -> add_extra_closure_args__exp(map,exp) end)}
            extra_args ->
                   new_args = Enum.map(extra_args,fn p -> {p, [], nil} end)
                   {fun, info, Enum.map(args,fn(exp) -> add_extra_closure_args__exp(map,exp) end) ++ new_args}
          end
       number when is_integer(number) or is_float(number) -> raise "Error: number is a command"
       {str,i1 ,a } -> {str,i1 ,a }

   end
end

defp add_extra_closure_args__if(map,[bexp, [do: then]]) do
 [add_extra_closure_args__exp(map,bexp), [do: add_extra_closure_args__body(map,then)]]
end
defp add_extra_closure_args__if(map,[bexp, [do: thenbranch, else: elsebranch]]) do
 [add_extra_closure_args__exp(map,bexp), [do: add_extra_closure_args__body(map,thenbranch), else: add_extra_closure_args__body(map,elsebranch)]]
end


defp add_extra_closure_args__exp(map,exp) do
   case exp do
     {{:., i1, [Access, :get]}, i2, [arg1,arg2]} ->
         {{:., i1, [Access, :get]}, i2, [arg1, add_extra_closure_args__exp(map,arg2)]}
     {{:., i1, [{struct, i2, nil}, field]},i3,[]} ->
         {{:., i1, [{struct, i2, nil}, field]},i3,[]}
     {{:., i1, [{:__aliases__, i2, [struct]}, field]}, i3, []} ->
       {{:., i1, [{:__aliases__, i2, [struct]}, field]}, i3, []}
     {op,info, args} when op in [:+, :-, :/, :*] ->
       {op,info, Enum.map(args, fn e -> add_extra_closure_args__exp(map,e) end)}
     {op, info, args} when op in [ :<=, :<, :>, :>=, :&&, :||, :!,:!=,:==] ->
       {op,info, Enum.map(args, fn e -> add_extra_closure_args__exp(map,e) end)}
     {var,info, nil} when is_atom(var) ->
       {var, info, nil}
     {fun,info, args} ->
      case map[fun] do
        nil -> {fun, info, Enum.map(args,fn(exp) -> add_extra_closure_args__exp(map,exp) end)}
        extra_args ->
            new_args = Enum.map(extra_args,fn p -> {p, [], nil} end)
           {fun, info, Enum.map(args,fn(exp) -> add_extra_closure_args__exp(map,exp) end) ++ new_args}
      end
     float when  is_float(float) -> float
     int   when  is_integer(int) -> int
     string when is_binary(string)  -> string
   end

 end

##################################
############# END closure elimination
###################################

##################################################
######################### OLD
######
##########################################
def compile_and_load_kernel({:ker, _k, k_type,{ast, is_typed?, delta}},  l) do

 # get the formal parameters of the function

  formal_par = get_args(ast)

 # get types of parameters:

  {:unit, type} = k_type


  # creates a map with the names that must be substituted (all parameters that are functions)


  map = create_map_subs(type, formal_par, l, %{})

 #  removes the arguments that will be substituted from the kernel definition

 ast = remove_args(map,ast)

 # makes the substitutions:

  ast = subs(map, ast)


  r = gen_jit_kernel_load(ast, is_typed?, delta)
  r
end

def gen_jit_kernel_load({:defk,_,[header,[body]]}, is_typed, inf_types) do

  {kname, _, para} = header

  param_list = para
       |> Enum.map(fn {p, _, _}-> PolyHok.CudaBackend.gen_para(p,Map.get(inf_types,p)) end)
       |> Enum.join(", ")

  types_para = para
       |>  Enum.map(fn {p, _, _}-> Map.get(inf_types,p) end)


  fname = "ker_" <> PolyHok.CudaBackend.gen_lambda_name()
  #fname = "k072b2a4iad"
  #fname = PolyHok.CudaBackend.gen_lambda_name()
  cuda_body = PolyHok.CudaBackend.gen_cuda(body,inf_types,is_typed,"")
  k = PolyHok.CudaBackend.gen_kernel(fname,param_list,cuda_body)
  accessfunc = PolyHok.CudaBackend.gen_kernel_call(fname,length(para),Enum.reverse(types_para))
  code = "\n" <> k <> "\n\n" <> accessfunc

 # IO.puts code
  file = File.open!("c_src/Elixir.App.cu", [:append])
  IO.write(file, "//#############################\n\n" <> code)
  File.close(file)
  {result, errcode} = System.cmd("nvcc",
        [ "--shared",
          "--compiler-options",
          "'-fPIC'",
          "-o",
          "priv/Elixir.App.so",
          "c_src/Elixir.App.cu"
    ], stderr_to_stdout: true)


    if ((errcode == 1) || (errcode ==2)) do raise "Error when JIT compiling .cu file generated by PolyHok: #{kname}\n #{result}" end
    IO.puts "antes"
    r = PolyHok.load_kernel_nif(to_charlist("Elixir.App"),to_charlist("#{fname}"))
    IO.puts "depois"
    #PolyHok.load_kernel_nif(to_charlist("Elixir.App"),to_charlist("map_kernel"))
    #PolyHok.load_fun_nif(to_charlist("Elixir.App"),to_charlist("#{fname}_call"))
    r
end
############## Removing from kernel definition the arguments that are functions
def remove_args(map, ast) do
   case ast do
        {:defk, info,[ {name, i2,  args} ,block]} ->  {:defk, info,[ {name, i2, filter_args(map,args)} ,block]}
        _ -> raise "Recompiling kernel: unknown ast!"
   end

end

def filter_args(map,[{var,i, nil}| t]) do
  if map[var] ==  nil do
    [{var,i, nil}| filter_args(map,t)]
  else
    filter_args(map,t)
  end
end
def filter_args(_map,[]), do: []

def get_args(ast) do
  case ast do
       {:defk, _info,[ {_name, _i2,  args} ,_block]} ->  args
       _ -> raise "Recompiling kernel: unknown ast!"
  end

end

#######################
#########
######### Creates a map with the substitutions to be made: formal parameter => actual paramenter
########
#######################
def create_map_subs([{_rt, funct} |tt], [{fname,_,nil} | tfa], [{:func, func, _type} | taa], map) when is_list(funct) and is_function(func) do
  case Macro.escape(func) do
    {:&, [],[{:/, [], [{{:., [], [_module, func_name]}, [no_parens: true], []}, _nargs]}]} ->
        create_map_subs(tt,tfa,taa,Map.put(map,fname,func_name))
    _ -> raise "Problem with paramenter #{inspect func}"

  end
end
def create_map_subs([_funct |tt], [{fname,_,nil} | tfa], [func | taa], map) when   is_function(func) do
  case Macro.escape(func) do
    {:&, [],[{:/, [], [{{:., [], [_module, func_name]}, [no_parens: true], []}, _nargs]}]} ->
        create_map_subs(tt,tfa,taa,Map.put(map,fname,func_name))
    _ -> raise "Problem with paramenter #{inspect func}"

  end
end
def create_map_subs([_funct |tt], [{fname,_,nil} | tfa], [{:anon, lambda,_type} | taa], map)  do
         # IO.inspect "yoooooo"
          #raise "hell"
          create_map_subs(tt,tfa,taa,Map.put(map,fname,lambda))
end
def create_map_subs([_t |tt], [_fa | tfa], [_aa | taa], map)  do
    create_map_subs(tt,tfa,taa,map)
end
def create_map_subs([], [], [], map), do: map
def create_map_subs(_,_,_,_), do: raise "spawn: wrong number of parameters at kernel launch."

###################
################### substitute variables that represent functions by the actual function names
############   (substitutes formal parameters that are functions by their actual values)
#### Takes the map created with create_map_subs and the ast and returns a new ast
########################

def subs(map,{:defk, i1,[header, [body]]}) do
   nbody = subs_body(map,body)
   {:defk, i1,[header, [nbody]]}
end


def subs_body(map,body) do


  case body do

      {:__block__, _, _code} ->
        subs_block(map,body)
      {:do, {:__block__,pos, code}} ->
        {:do, subs_block(map, {:__block__, pos,code}) }
      {:do, exp} ->
        {:do, subs_command(map,exp)}
      {_,_,_} ->
        subs_command(map,body)
   end


end
defp subs_block(map,{:__block__, info, code}) do
  {:__block__, info,
      Enum.map(code,  fn com -> subs_command(map,com) end)
  }
end

defp subs_command(map,code) do
    case code do
        {:for,i,[param,[body]]} ->
          {:for,i,[param,[subs_body(map,body)]]}
        {:do_while, i, [[doblock]]} ->
          {:do_while, i, [[subs_body(map,doblock)]]}
        {:do_while_test, i, [exp]} ->
          {:do_while_test, i, [subs_exp(map,exp)]}
        {:while, i, [bexp,[body]]} ->
          {:while, i, [subs_exp(map,bexp),[subs_body(map,body)]]}
        # CRIAÇÃO DE NOVOS VETORES
        {{:., i1, [Access, :get]}, i2, [arg1,arg2]} ->
          {{:., i1, [Access, :get]}, i2, [subs_exp(map,arg1),subs_exp(map,arg2)]}
        {:__shared__, i1, [{{:., i2, [Access, :get]}, i3, [arg1,arg2]}]} ->
          {:__shared__,i1 , [{{:., i2, [Access, :get]}, i3, [subs_exp(map,arg1),subs_exp(map,arg2)]}]}

        # assignment
        {:=, i1, [{{:., i2, [Access, :get]}, i3, [{array,a1,a2},acc_exp]}, exp]} ->
          {:=, i1, [{{:., i2, [Access, :get]}, i3, [{array,a1,a2},subs_exp(map,acc_exp)]}, subs_exp(map,exp)]}
        {:=, i, [var, exp]} ->
          {:=, i, [var, subs_exp(map,exp)]}
        {:if, i, if_com} ->
          {:if, i, subs_if(map,if_com)}
        {:var, i1 , [{var,i2,[{:=, i3, [{type,ii,nil}, exp]}]}]} ->
          {:var, i1 , [{var,i2,[{:=, i3, [{type,ii,nil}, subs_exp(map,exp)]}]}]}
        {:var, i1 , [{var,i2,[{:=, i3, [type, exp]}]}]} ->
          {:var, i1 , [{var,i2,[{:=, i3, [type, subs_exp(map,exp)]}]}]}
        {:var, i1 , [{var,i2,[{type,i3,t}]}]} ->
          {:var, i1 , [{var,i2,[{type,i3,t}]}]}
        {:var, i1 , [{var,i2,[type]}]} ->
          {:var, i1 , [{var,i2,[type]}]}
        {:type, i1 , [{var,i2,[{type,i3,t}]}]} ->
          {:type, i1 , [{var,i2,[{type,i3,t}]}]}
        {:type, i1 , [{var,i2,[type]}]} ->
          {:type, i1 , [{var,i2,[type]}]}

        {:return,i,[arg]} ->
          {:return,i,[subs_exp(map,arg)]}

        {fun, info, args} when is_list(args)->
          new_name = map[fun]
          if (new_name == nil ) do
            {fun, info, Enum.map(args,fn(exp) -> subs_exp(map,exp) end)}
          else
            {new_name, info, Enum.map(args,fn(exp) -> subs_exp(map,exp) end)}
          end
        number when is_integer(number) or is_float(number) -> raise "Error: number is a command"
        {str,i1 ,a } -> {str,i1 ,a }

    end
end

defp subs_if(map,[bexp, [do: then]]) do
  [subs_exp(map,bexp), [do: subs_body(map,then)]]
end
defp subs_if(map,[bexp, [do: thenbranch, else: elsebranch]]) do
  [subs_exp(map,bexp), [do: subs_body(map,thenbranch), else: subs_body(map,elsebranch)]]
end


defp subs_exp(map,exp) do
    case exp do
      {{:., i1, [Access, :get]}, i2, [arg1,arg2]} ->
          {{:., i1, [Access, :get]}, i2, [arg1, subs_exp(map,arg2)]}
      {{:., i1, [{struct, i2, nil}, field]},i3,[]} ->
          {{:., i1, [{struct, i2, nil}, field]},i3,[]}
      {{:., i1, [{:__aliases__, i2, [struct]}, field]}, i3, []} ->
        {{:., i1, [{:__aliases__, i2, [struct]}, field]}, i3, []}
      {op,info, args} when op in [:+, :-, :/, :*] ->
        {op,info, Enum.map(args, fn e -> subs_exp(map,e) end)}
      {op, info, args} when op in [ :<=, :<, :>, :>=, :&&, :||, :!,:!=,:==] ->
        {op,info, Enum.map(args, fn e -> subs_exp(map,e) end)}
      {var,info, nil} when is_atom(var) ->
        {var, info, nil}
      {fun,info, args} ->
        new_name = map[fun]
        if (new_name == nil ) do
          {fun, info, Enum.map(args,fn(exp) -> subs_exp(map,exp) end)}
        else
          {new_name, info, Enum.map(args,fn(exp) -> subs_exp(map,exp) end)}
        end
      float when  is_float(float) -> float
      int   when  is_integer(int) -> int
      string when is_binary(string)  -> string
    end

  end


end
