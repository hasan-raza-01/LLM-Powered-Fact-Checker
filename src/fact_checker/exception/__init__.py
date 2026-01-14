import sys as SYS

from src.fact_checker import logging





class CustomException(Exception):



    def __init__(self, message:str | Exception, sys:SYS):

        self.message = message

        super().__init__(message)

        _, _, exc_traceback = sys.exc_info()

        self.path = exc_traceback.tb_frame.f_code.co_filename

        self.line = exc_traceback.tb_lineno



    def __str__(self):

        return f"CustomException: {self.message} on line: {self.line} of {self.path}"



if __name__=="__main__":

    try:

        1/0

    except Exception as e:

        logging.exception(e)

        raise CustomException(e, SYS)



