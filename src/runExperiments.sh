#!/bin/python3

import sys
import os

def was_executed(exp, executed_exps):
    
    for line in executed_exps:
        if exp == line:
            return True
    
    return False
    

if len(sys.argv) != 2:
   print("Run:",sys.argv[0], "<experiments_file>")
   sys.exit(0)

python_interpreter = "/home/carlos/Informatica/Investigacion/MisEstudios/PonyGE2/venv/bin/python"
path_to_program = '/home/carlos/Informatica/Investigacion/MisEstudios/PonyGE2/src/scripts/'

hubo_ejecucion = True
executed_filename = sys.argv[1] + "_executed"
os.system("touch " + executed_filename)

while hubo_ejecucion:
   hubo_ejecucion = False

   a_file = open(sys.argv[1], 'r')
   lines = a_file.readlines()
   executed_file = open(executed_filename, 'r')
   executed_exps = executed_file.readlines()
   executed_file.close()
   line = ''

   for line in lines:
      if len(line) > 0 and not line.startswith('#') and not was_executed(line, executed_exps):

         try:
            print(line)
            os.system(python_interpreter + " " + path_to_program + line)
         except:
            executed_file = open(executed_filename, 'a')
            executed_file.write("Error con " + line)
            executed_file.close()

         executed_file = open(executed_filename, 'a')
         executed_file.write(line)
         executed_file.close()
         
         hubo_ejecucion = True
         break

   a_file.close()
    

    
