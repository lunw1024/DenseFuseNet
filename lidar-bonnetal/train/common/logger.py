# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
# MODIFIED: rewritten using torch.utils.tensorboard

import torch.utils.tensorboard as tensorboard
import torchvision
import numpy as np


class Logger(object):

  def __init__(self, log_dir):
    """Create a summary writer logging to log_dir."""
    self.writer = tensorboard.SummaryWriter(log_dir)

  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    self.writer.add_scalar(tag, value, step)
    self.writer.flush()

  def image_summary(self, tag, images, step):
    """Log a list of images."""
    print("Trying to log images. This wasn't tested, be careful!")
    img_grid = torchvision.utils.make_grid(images)
    self.writer.add_images(tag, img_grid, step)
    self.writer.flush()

  def histo_summary(self, tag, values, step, bins=1000):
    """Log a histogram of the tensor of values."""
    # Create and write Summary
    self.writer.add_histogram(tag, values, step, bins=bins)
    self.writer.flush()
