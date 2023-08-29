import sys

from std_srvs.srv import Trigger
import rclpy
from rclpy.node import Node
import  threading
import os
import yaml
import  time

class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(Trigger, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Trigger.Request()

    def send_request(self):
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main(args=None):
    rclpy.init(args=args)
    yaml_result_file = "/total_route.yaml"

    route_dict = {}
    if os.path.exists(yaml_result_file):
        with open(yaml_result_file, 'r') as stream:
            route_dict = yaml.safe_load(stream)

    max_len = 0
    for agent in route_dict:
        route_len = len(route_dict[agent])
        max_len = max(max_len, route_len)

    time_sequence = [0.1] * 1000
    time_sequence.insert(0, 5)
    minimal_client = MinimalClientAsync()
    #
    # thread = threading.Thread(target=rclpy.spin, args=(minimal_client,), daemon=True)
    # thread.start()

    for timer in time_sequence:
        # rclpy.spin_once(minimal_client)
        response = minimal_client.send_request()
        # rate = minimal_client.create_rate(timer*0.1, minimal_client.get_clock())
        print('sleep time - ', timer)

        minimal_client.get_logger().info(
            'Result of the client call:  %d' %
            (response.success))

        time.sleep(timer)

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()