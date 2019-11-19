"""
    Doc: The following class will create the behaviour of a ring buffer. It is composed by the following methods:
        - insert_element(): overwrites the insert method of list, does not overwrites elements, but, if it detects that the 
                    list is occupied at that index it will pop the element, making it save to insert the data. Also, 
                    it differentiates between new data and retransmitted data, updating the written sequence to check it.
        
        - get_iterator(): returns the elements of the list from the last read index position.
        
        - consume_elements(): it will consume elements from the read index up to the number specified, it will update
                              the read index position to the last consumed element.

"""
import bisect
from collections import namedtuple
from operator import or_, and_

import math
from debug_log import *

class ring_buffer(list):
    class Item(namedtuple('Item',('seq', 'end', 'data'))):
        def __gt__(self, other):
            return self[0:2] > other[0:2]

        def __lt__(self, other):
            return self[0:2] < other[0:2]

        def __eq__(self, other):
            return self[0:2] == other[0:2]


    def __init__(self, size):
        list.__init__(self)
        self.size = size

        self.read_idx = 0
        self.read_seq = 0

        self.tail_seq = 0


    # Insert only the data that does not overlap
    def insert_element(self, seq, length, data):
        if seq > self.size:
            raise IndexError("Sequence number can not be more than RB size")

        end = (seq+length) % self.size

        elem = self.Item(seq, end, data)
        l=len(self)
        #If buffer empty just write value
        if l == 0 and self.tail_seq == self.read_seq:
             debug("inserting into empty")
             list.insert(self, 0, elem)
             self.read_idx = 0
             self.tail_seq = end
             return
        #else figure where its head lands
        idx = bisect.bisect_left(self, elem)

        #Check where the tail would land, we want it to be the same position, clean up if needed
        tmp = self.Item(end, end, None)
        idx2 = bisect.bisect_left(self, tmp)
        debug("new item {}, idx {} idx2 {}".format(elem, idx,idx2))
        while idx2 > idx:  # TODO: NOT SURE ABOUT THIS ONE YET... (PROBABLY NEEDS A FIX)
            assert idx2 >= idx
            debug("target idx {} popping {}".format(idx, idx2-1))
            x=list.pop(self, idx2-1)
            if self.read_idx == list.__len__(self):
                debug("Updated read idx=0")
                self.read_idx = 0
            self.read_seq = x.end
            idx2 = bisect.bisect_left(self, tmp)

        # update read index and insert the element
        if list.__len__(self) > 0 and self.read_seq > seq:
            self.read_idx += 1
        list.insert(self, idx, elem)
        #See if our read and write points  are "inverted"
        inv = self.read_seq > self.tail_seq
        #Write the conditions for inclusions into ranges...
        cond1 = [and_, or_][inv]
        cond2 = [and_, or_][not inv]

        #Find where our data lands in terms of seq
        if cond2(self.read_seq > seq, seq >= self.tail_seq):
            #Between tail seq and read sequence -> New data
            debug("New data with seqNum = {}, inserted at {}",(seq, idx))
            self.tail_seq = end
        elif cond1(self.read_seq <= seq, seq < self.tail_seq):
            # Between read seq and tail sequence -> Retransmitted data
            debug("Retransmission, seqNum = {}, tail {} read {}, inserted at {}",(
                seq, self.tail_seq, self.read_seq, idx))
        else:
            raise ValueError("THIS IS IMPOSSIBLE")
        try:
            self.__getitem__(self.read_idx)
        except:
            print(self.read_idx)
            critical("Invalid read index transition!!!")
            exit()



    def available_length(self):
        """ Calculates the available length in the ring buffer"""
        debug("Read Seq {}, Tail Seq {}", [self.read_seq, self.tail_seq])
        if self.read_seq == self.tail_seq:
            return self.size
        elif self.read_seq > self.tail_seq:
            return self.read_seq - self.tail_seq
        else:
            return self.read_seq + (self.size - self.tail_seq)


    def get_iterator(self):
        """:returns iterator that provides elements from read position onwards"""
        L = len(self)
        if L == 0:
            raise StopIteration("Empty list")

        if self.read_seq != self[self.read_idx][0]:
            debug("Stucked: expected seq = {}; next seq = {}; index = {}; size of the ring {}\n",
                  [self.read_seq, self[self.read_idx][0], self.read_idx, self.size])
            raise StopIteration("Missing sequence")

        p = self.read_idx
        while True:
            element = self[p]
            p = (p + 1) % L
            yield element
            if p == self.read_idx:
                break
        raise StopIteration("No more entries")


    def ensure_free_space(self, seq_amount):
        """
        Consumes given number of sequence numbers
        Extracts at least seq_amount of bytes from the ring
        """
        to_extract = []
        cur_len = self.available_length()
        if cur_len < seq_amount:
            #trim the oldest seq
            if list.__len__(self) > 0:
                self.read_seq = self[self.read_idx].seq
            else:
                self.read_seq = self.tail_seq
            cur_len = self.available_length()
        while cur_len<seq_amount and list.__len__(self) > 0:
            next_frag = self[self.read_idx]
            if next_frag.end > next_frag.seq:
                size = next_frag.end - next_frag.seq
            else:
                size = self.size - next_frag.seq + next_frag.end
            cur_len  = self.available_length()
            to_extract.append(next_frag)
            self.consume_by_number(1)
        debug("cleaned {} entries", len(to_extract))
        return to_extract

    def consume_by_number(self, num):
        """consumes elements at read position up to num"""
        assert num <= len(self)
        while num>0 and list.__len__(self) > 0:
            num -=1
            self.read_seq = list.__getitem__(self, self.read_idx).end
            list.pop(self, self.read_idx)
            if self.read_idx == len(self):# if we have popped the last element of the array
                #start from beginning
                self.read_idx = 0



    def __str__(self):
        s = "RING: Size={} Len={}, RI={}, RS={}, TS={}\n".format(self.size, len(self), self.read_idx,
                                                                 self.read_seq, self.tail_seq)
        for i in self.get_iterator():
            s += str(i)+"\n"
        return s


if __name__ == "__main__":
    import random
    set_debug(0)
    N = 100
    rb = ring_buffer(N)
    # rb.insert_element(10, 20, 'E {}'.format(10))
    # rb.insert_element(30, 20, 'E {}'.format(30))
    # rb.consume_by_number(1)
    # rb.insert_element(50, 40, 'E {}'.format(50))
    # rb.insert_element(90, 10, 'E {}'.format(90))
    # print(1,rb)
    # rb.consume_by_number(1)
    # print(2,rb)
    # rb.insert_element(30, 20, 'E {}'.format(30))
    # print(3,rb)

    rb = ring_buffer(N)
    rb.insert_element(10, 20, 'E {}'.format(10))
    rb.insert_element(10, 20, 'E {}'.format(10))
    print(rb)
    exit()
    rb.consume_by_number(1)

    rb.insert_element(50, 40, 'E {}'.format(50))
    rb.insert_element(30, 20, 'E {}'.format(30))
    print(rb)
    exit()
    random.seed(1)
    N=100
    rb = ring_buffer(N)
    seq = 10
    for t in range(10000):
        sz = random.randint(5,int(N*0.24))
        if random.uniform(0, 1) < 0.9:
            rb.insert_element(seq, sz, "E {}".format(seq))
        seq = (seq + sz) % N

        if random.uniform(0,1) > 0.99 and len(rb)//2>1:
            rb.consume_by_number(random.randint(1, len(rb)//2))
        rb.ensure_free_space(int(N*0.25))

    for i in rb.get_iterator():
        print(i)
    print(rb)
