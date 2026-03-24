"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""
import queue

import numpy as np
from gnuradio import gr
import time
import sys
import shared_data

class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """Embedded Python Block example - a simple multiply const"""

    def __init__(self):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='timestamp_csi',   # will show up in GRC
            in_sig=[(np.complex64, 52)],
            out_sig=None
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).

    def work(self, input_items, output_items):
        """example: multiply with constant"""
        in_data = input_items[0]
        n_vectors = len(in_data)

        if n_vectors > 0:
            current_time = time.time()

            # 遍历二维数组，按行提取每一个 CSI 包
            for i in range(n_vectors):
                # if 'shared_data' in sys.modules:
                shared_data.csi_queue.put((current_time,in_data[i].copy()))
        # 告诉 GNU Radio 我们消耗了 N 个向量
        return n_vectors