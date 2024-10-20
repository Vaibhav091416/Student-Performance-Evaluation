#to handle exceptions
import sys
import logging
from src.logger import logging

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    #it'll tell which file and whih lines exception occur
    file_name=exc_tb.tb_frame.f_code.co_filename
    #from custom error docs of python
    error_message="Error Occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message

#note this class is being inherited from already existing Exception class of python
class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):  #constructor
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message
    
# if __name__=='__main__':
#     try:
#         a=1/0
#     except Exception as e:
#         logging.info("Devide by zero error")
#         raise CustomException(e,sys)
