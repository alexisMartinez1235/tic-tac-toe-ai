import debugpy
debugpy.listen(("localhost", 5678))

while True:
  debugpy.wait_for_client()
  while True:
    debugpy.breakpoint() 
