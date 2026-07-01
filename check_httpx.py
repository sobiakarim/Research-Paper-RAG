import httpx
import inspect
print('httpx', httpx.__version__)
print('signature:', inspect.signature(httpx.Client.__init__))
