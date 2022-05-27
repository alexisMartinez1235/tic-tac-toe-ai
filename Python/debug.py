import debugpy
debugpy.listen(("localhost", 5678))

while True:
  debugpy.wait_for_client()
  debugpy.breakpoint()
  debugpy.log_to("output.txt")
