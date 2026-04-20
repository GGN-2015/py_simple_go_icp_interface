import cpp_simple_interface
import os

try:
    from .utils import GO_ICP_EXE, GO_ICP_CPP_LIST
except:
    from utils import GO_ICP_EXE, GO_ICP_CPP_LIST

def compile_go_icp(force:bool=False) -> bool:
    if force:
        if os.path.isfile(GO_ICP_EXE):
            os.remove(GO_ICP_EXE)
    
    # find exe
    if os.path.isfile(GO_ICP_EXE):
        return True
    
    # recompile
    print("Compiling Go-ICP ...")
    suc, msg = cpp_simple_interface.compile_cpp_files(
        GO_ICP_CPP_LIST, 
        GO_ICP_EXE)
    if not suc:
        print(msg)
        raise RuntimeError("Compilation error")
    return suc

if __name__ == "__main__":
    compile_go_icp()
