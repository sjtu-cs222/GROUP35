from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from models.msdnet_layers import MSDLayer,\
    MSDFirstLayer,CifarClassifier,Transition
from torch.autograd import Variable
import math

__all__ = ['MSDNet']


class MSDNet(nn.Module):

    def __init__(self, args):
        """
        The main module for Multi Scale Dense Network.
        It holds the different blocks with layers and classifiers of the MSDNet layers

        :param args: Network argument
        """

        super(MSDNet, self).__init__()

        # Init arguments
        self.args = args
        
        self.base = self.args.msd_base
        self.step = self.args.msd_step
        self.step_mode = self.args.msd_stepmode
        self.msd_prune = self.args.msd_prune
        self.num_blocks = self.args.msd_blocks
        self.reduction_rate = self.args.reduction
        self.growth = self.args.msd_growth
        self.growth_factor = args.msd_growth_factor
        self.bottleneck = self.args.msd_bottleneck
        self.bottleneck_factor = args.msd_bottleneck_factor
        self.gradients = []

        try:
            self.maxC = self.args.maxC
        except:
            self.maxC = self.num_blocks
        

        # Set progress
        if args.data in ['cifar10', 'cifar100']:
            self.image_channels = 3
            self.num_channels = 32
            self.num_scales = 3
            self.num_classes = int(args.data.strip('cifar'))
        else:
            raise NotImplementedError

        # Init MultiScale graph and fill with Blocks and Classifiers
        print('| MSDNet-Block {}-{}-{}'.format(self.num_blocks,
                                               self.step,
                                               self.args.data))
        (self.num_layers, self.steps) = self.calc_steps()

        print('Building network with the steps: {}'.format(self.steps))
        self.cur_layer = 1
        self.cur_transition_layer = 1
        self.subnets = nn.ModuleList(self.build_modules(self.num_channels))

        if args.maxC == 1:
            self.subnets[0].MSD_layer_4.subnets[-1].register_backward_hook(self.hookFunc)
        elif args.maxC == 2:
            self.subnets[1].MSD_layer_6.subnets[-1].register_backward_hook(self.hookFunc)
        elif args.maxC == 3:        
            self.subnets[2].MSD_layer_8.subnets[-1].register_backward_hook(self.hookFunc)
        elif args.maxC == 4:
            self.subnets[3].MSD_layer_10.subnets[-1].register_backward_hook(self.hookFunc)
        elif args.maxC == 5:
            self.subnets[4].MSD_layer_12.subnets[-1].register_backward_hook(self.hookFunc)
        elif args.maxC == 6:
            self.subnets[5].MSD_layer_14.subnets[-1].register_backward_hook(self.hookFunc)
        elif args.maxC == 7:
            self.subnets[6].MSD_layer_16.subnets[-1].register_backward_hook(self.hookFunc)
        elif args.maxC == 8:
            self.subnets[7].MSD_layer_18.subnets[-1].register_backward_hook(self.hookFunc)
        elif args.maxC == 9:
            self.subnets[8].MSD_layer_20.subnets[-1].register_backward_hook(self.hookFunc)
        elif args.maxC == 10:
            self.subnets[9].MSD_layer_22.subnets[-1].register_backward_hook(self.hookFunc)

        for m in self.subnets:
            self.init_weights(m)
            if hasattr(m,'__iter__'):
                for sub_m in m:
                    self.init_weights(sub_m)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def calc_steps(self):
        """Calculates the number of layers required in each
        Block and the total number of layers, according to
        the step and stepmod.

        :return: number of total layers and list of layers/steps per blocks
        """

        # Init steps array
        steps = [None]*self.num_blocks
        steps[0] = num_layers = self.base

        # Fill steps and num_layers
        for i in range(1, self.num_blocks):

            # Take even steps or calc next linear growth of a step
            steps[i] = (self.step_mode == 'even' and self.step) or \
                        self.step*(i-1)+1
            num_layers += steps[i]

        return num_layers, steps

    def build_modules(self, num_channels):
        """Builds all blocks and classifiers and add it
        into an array in the order of the format:
        [[block]*num_blocks [classifier]*num_blocks]
        where the i'th block corresponds to the (i+num_block) classifier.

        :param num_channels: number of input channels
        :return: An array with all blocks and classifiers
        """

        # Init the blocks & classifiers data structure
        modules = [None] * self.num_blocks * 2
        for i in range(0, self.num_blocks):
            print ('|-----------------Block {:0>2d}----------------|'.format(i+1))

            # Add block
            modules[i], num_channels = self.create_block(num_channels, i)

            # Calculate the last scale (smallest) channels size
            channels_in_last_layer = num_channels *\
                                     self.growth_factor[self.num_scales]

            # Add a classifier that belongs to the i'th block
            modules[i + self.num_blocks] = \
                CifarClassifier(channels_in_last_layer, self.num_classes)
        return modules

    def create_block(self, num_channels, block_num):
        '''
        :param num_channels: number of input channels to the block
        :param block_num: the number of the block (among all blocks)
        :return: A sequential container with steps[block_num] MSD layers
        '''

        block = nn.Sequential()

        # Add the first layer if needed
        if block_num == 0:
            block.add_module('MSD_first', MSDFirstLayer(self.image_channels,
                                                        num_channels,
                                                        self.num_scales,
                                                        self.args))

        # Add regular layers
        current_channels = num_channels
        for _ in range(0, self.steps[block_num]):

            # Calculate in and out scales of the layer (use paper heuristics)
            if self.msd_prune == 'max':
                interval = math.ceil(self.num_layers/
                                      self.num_scales)
                in_scales = int(self.num_scales - \
                            math.floor((max(0, self.cur_layer - 2))/interval))
                out_scales = int(self.num_scales - \
                             math.floor((self.cur_layer - 1)/interval))
            else:
                raise NotImplementedError

            self.print_layer(in_scales, out_scales)
            self.cur_layer += 1

            # Add an MSD layer
            block.add_module('MSD_layer_{}'.format(self.cur_layer - 1),
                             MSDLayer(current_channels,
                                      self.growth,
                                      in_scales,
                                      out_scales,
                                      self.num_scales,
                                      self.args))

            # Increase number of channel (as in densenet pattern)
            current_channels += self.growth

            # Add a transition layer if required
            if (self.msd_prune == 'max' and in_scales > out_scales and
                self.reduction_rate):

                # Calculate scales transition and add a Transition layer
                offset = self.num_scales - out_scales
                new_channels = int(math.floor(current_channels*
                                              self.reduction_rate))
                block.add_module('Transition', Transition(
                    current_channels, new_channels, out_scales,
                    offset, self.growth_factor, self.args))
                print('|      Transition layer {} was added!      |'.
                      format(self.cur_transition_layer))
                current_channels = new_channels

                # Increment counters
                self.cur_transition_layer += 1

            elif self.msd_prune != 'max':
                raise NotImplementedError

        return block, current_channels

    def print_layer(self, in_scales, out_scales):
        print('| Layer {:0>2d} input scales {} output scales {} |'.
              format(self.cur_layer, in_scales, out_scales))

    def save_gradient(self, grad):
    	self.gradients.append(grad)
    
    def hookFunc(self, module, gradInput, gradOutput):
        self.gradients.append(gradOutput[0])
        # self.gradients.append(gradOutput[0])

        # import pdb
        # pdb.set_trace()
    def forward(self, x, progress=None, p=0,demand=-1):
        """
        Propagate Input image in all blocks of MSD layers and classifiers
        and return a list of classifications

        :param x: Input image / batch
        :return: a list of classification outputs
        """

        outputs = [None] * self.num_blocks
        feats = [None] * self.num_blocks
        probab = [None] * self.num_blocks

        outputs = [None] * self.maxC
        feats = [None] * self.maxC
        probab = [None] * self.maxC


        cur_input = x
        # for block_num in range(0, self.num_blocks):
        if demand==-2:
            for block_num in range(0, self.maxC ):
                block = self.subnets[block_num]
                cur_input = block_output = block(cur_input)
                class_output = self.subnets[block_num+self.num_blocks](block_output[-1])
                feat = block_output[-1]
                outputs[block_num] = class_output
                probab[block_num] = torch.nn.functional.softmax(class_output, dim=-1)
                if torch.max(probab[block_num],1)[0].cpu().item()>0.5:
                    return outputs[block_num],probab
                feats[block_num] = feat
                
        if demand!=-1:
            for block_num in range(demand+1):
                block = self.subnets[block_num]
                cur_input = block_output = block(cur_input)
            class_output = self.subnets[demand+self.num_blocks](block_output[-1])
            feat = block_output[-1]
            outputs[demand] = class_output
            probab[demand] = torch.nn.functional.softmax(class_output, dim=-1)
            feats[block_num] = feat
            return outputs[demand],probab
        for block_num in range(0, self.maxC ):

            # Get the current block's output
            if self.args.debug:
                print("")
                print("Forwarding to block %s:" % str(block_num + 1))
            block = self.subnets[block_num]
            cur_input = block_output = block(cur_input)

            # Classify and add current output
            if self.args.debug:
                print("- Getting %s block's output" % str(block_num + 1))
                for s, b in enumerate(block_output):
                    print("- Output size of this block's scale {}: ".format(s),
                          b.size())
            class_output = self.subnets[block_num+self.num_blocks](block_output[-1])
            feat = block_output[-1]
            outputs[block_num] = class_output
            probab[block_num] = torch.nn.functional.softmax(class_output, dim=-1)
            feats[block_num] = feat

        if p==0:
            return outputs
        if p==1:
            return outputs, feats
        else:
            return outputs, probab

