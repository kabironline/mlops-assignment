-- This script tells wrk how to format the POST request

-- Set method, body, and headers for the request
wrk.method = "POST"
wrk.body   = '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
wrk.headers["Content-Type"] = "application/json"

-- Optional: You could randomize the body data here for a more realistic test
-- function request()
--   local sepal_l = string.format("%.1f", math.random(40, 80) / 10)
--   local sepal_w = string.format("%.1f", math.random(20, 50) / 10)
--   local petal_l = string.format("%.1f", math.random(10, 70) / 10)
--   local petal_w = string.format("%.1f", math.random(1, 30) / 10)
--
--   wrk.body = '{"sepal_length": '..sepal_l..', "sepal_width": '..sepal_w..', "petal_length": '..petal_l..', "petal_width": '..petal_w..'}'
--   return wrk.request()
-- end