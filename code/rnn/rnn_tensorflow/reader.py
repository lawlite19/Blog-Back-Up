import numpy as np


def ptb_iterator(raw_data, batch_size, num_steps, steps_ahead=1):
  """Iterate on the raw PTB data.
  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.
  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  raw_data = np.array(raw_data, dtype=np.int32)

  data_len = len(raw_data)

  batch_len = data_len // batch_size

  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  offset = 0
  if data_len % batch_size:
    offset = np.random.randint(0, data_len % batch_size)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i + offset:batch_len * (i + 1) + offset]

  epoch_size = (batch_len - steps_ahead) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+steps_ahead]
    yield (x, y)

  #if epoch_size * num_steps < batch_len - steps_ahead:
    #yield (data[:, epoch_size*num_steps : batch_len - steps_ahead], data[:, epoch_size*num_steps + 1:])

def ptb_iterator_oldversion(raw_data, batch_size, num_steps):
  """Iterate on the raw PTB data.
  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.
  Args:
  raw_data: one of the raw data outputs from ptb_raw_data.
  batch_size: int, the batch size.
  num_steps: int, the number of unrolls.
  Yields:
  Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
  The second element of the tuple is the same data time-shifted to the
  right by one.
  Raises:
  ValueError: if batch_size or num_steps are too high.
  """
  raw_data = np.array(raw_data, dtype=np.int32)
  
  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
  
  epoch_size = (batch_len - 1) // num_steps
  
  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
  
  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)
  
  

def shuffled_ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)
    r = len(raw_data) % num_steps
    if r:
        n = np.random.randint(0, r)
        raw_data = raw_data[n:n + len(raw_data) - r]

    raw_data = np.reshape(raw_data, [-1, num_steps])
    np.random.shuffle(raw_data)

    num_batches = int(np.ceil(len(raw_data) / batch_size))

    for i in range(num_batches):
        data = raw_data[i*batch_size:min(len(raw_data), (i+1)*batch_size),:]
        yield (data[:,:-1], data[:,1:])
        
