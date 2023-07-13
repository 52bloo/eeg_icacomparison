import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as nnF


class SELayer(nn.Module):
    """
    Squeeze and excitation layer used for MLSTM_FCN
    Todo: initializing fc_x with he_uniform (Kaming initialziation): where?
    """
    def __init__(self, n_channels, reduction=16):
        super().__init__()

        self.avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.fc_1 = nn.Linear(n_channels, n_channels // reduction, bias=False)
        self.fc_2 = nn.Linear(n_channels // reduction, n_channels, bias=False)

    def forward(self, x: torch.Tensor):
        d1, d2, d3 = x.size()
        x_int: torch.Tensor = self.avgpool1d(x).view(d1, d2)
        x_int = nnF.relu(self.fc_1(x_int))
        x_int = torch.sigmoid(self.fc_2(x_int)).view(d1, d2, 1)
        x_int = x * x_int.expand_as(x)
        return x_int

# as close to the reference paper as possible
class MLSTM_FCN_Ref(nn.Module):
    def __init__(self, input_size, max_sequence_size, output_classn,
                 lstm_hidden_size=8, lstm_num_layers=1, lstm_batch=True,
                 conv1_size=128, conv2_size=256, conv3_size=128,
                 conv1_kernel=8, conv2_kernel=5, conv3_kernel=3,
                 lstm_dropout=0.8, fc_dropout=0.3):
        '''

        :param input_size: EEG channel size
        :param max_sequence_size: Time sample size
        :param output_classn: self explanatory
        :param lstm_hidden_size:
        :param lstm_num_layers:
        :param lstm_batch:
        :param conv1_size: filter size for out/in
        :param conv2_size: filter size for out/in
        :param conv3_size: filter size for out/in
        :param conv1_kernel:
        :param conv2_kernel:
        :param conv3_kernel:
        :param lstm_dropout:
        :param fc_dropout:
        '''
        super().__init__()

        self.input_size = input_size
        self.max_sequence_size = max_sequence_size
        self.output_classn = output_classn
        # lstm parameters
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_batch = lstm_batch
        self.lstm_dropout = lstm_dropout
        # conv parameters
        self.conv1_size = conv1_size
        self.conv2_size = conv2_size
        self.conv3_size = conv3_size
        self.conv1_kernel = conv1_kernel
        self.conv2_kernel = conv2_kernel
        self.conv3_kernel = conv3_kernel
        # fc parameters
        self.fc_dropout = fc_dropout

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_num_layers, batch_first=self.lstm_batch)

        # arg 3 can be an int! ignore the tuple complaint by lint
        self.conv1 = nn.Conv1d(self.input_size, self.conv1_size, self.conv1_kernel)
        self.conv2 = nn.Conv1d(self.conv1_size, self.conv2_size, self.conv2_kernel)
        self.conv3 = nn.Conv1d(self.conv2_size, self.conv3_size, self.conv3_kernel)

        self.fc = nn.Linear(self.conv3_size + self.lstm_hidden_size, self.output_classn)

        self.bn1 = nn.BatchNorm1d(self.conv1_size)
        self.bn2 = nn.BatchNorm1d(self.conv2_size)
        self.bn3 = nn.BatchNorm1d(self.conv3_size)

        self.se1 = SELayer(self.conv1_size)
        self.se2 = SELayer(self.conv2_size)

        self.drop_lstm = nn.Dropout(self.lstm_dropout)
        #self.drop_fc = nn.Dropout(self.fc_dropout)

    def forward(self, x):
        """
        X should be Batch X time sample x EEG channel
        (wouldn't batch size be determined by other pytorch code?)
        Assumes equal length in sequences
        :param x:
        :param batch_tlength:
        :return:
        """
        # x1 = lstm pathway
        # x2 = conv1d pathway

        # packing sequence and feeding through lstm (dimension shuffle?)
        # pre packing x1 dimension would be Batch x Time x EEGchan
        # after packing x1 dimension would be (Batch x Time) x EEGchan
        #x1 = nn.utils.rnn.pack_padded_sequence(x, batch_tlength, batch_first=True, enforce_sorted=False)
        x1, (h_state, c_state) = self.lstm(x)
        x1 = self.drop_lstm(x1)

        # use the last timestep for the output
        # (Keras automatically puts out the last one while in torch you have to specify it)
        x1 = x1[:,-1,:] #.reshape([x1.shape[0], x1.shape[2], 1])

        # x1 = x1.transpose(2, 1)
        #x1, seq_len = nn.utils.rnn.pad_packed_sequence(x1, batch_first=True)
        # once it is padded again it should be returned to the original dimension
        # (since the batch_first argument is the same in both calls)



        # weights are initialized with kaming_uniform() by default in pytorch, no need to
        # specify explicit initialization
        x2 = x.transpose(2, 1)
        x2 = nnF.relu(self.bn1(self.conv1(x2)))
        x2 = self.se1(x2)
        x2 = nnF.relu(self.bn2(self.conv2(x2)))
        x2 = self.se2(x2)
        x2 = nnF.relu(self.bn3(self.conv3(x2)))
        # use global average pooling: check for dimension requirements
        x2 = nnF.adaptive_avg_pool1d(x2, 1)
        x2 = x2.reshape([x2.shape[0], x2.shape[1]])
        # x2 = torch.mean(x2, 2)

        # first dimension
        x_combined = torch.cat((x1, x2), dim=1)
        x_out = self.fc(x_combined)
        #x_out = nnF.log_softmax(x_out, dim=1) # CrossEntropyLoss contains log_softmax by default

        return x_out

class MLSTM_FCN_Varlength(nn.Module):
    def __init__(self, input_size, max_sequence_size, output_classn,
                 lstm_hidden_size=128, lstm_num_layers=1, lstm_batch=True,
                 conv1_size=128, conv2_size=256, conv3_size=128,
                 conv1_kernel=8, conv2_kernel=5, conv3_kernel=3,
                 lstm_dropout=0.8, fc_dropout=0.3):
        super().__init__()

        self.input_size = input_size
        self.max_sequence_size = max_sequence_size
        self.output_classn = output_classn
        # lstm parameters
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_batch = lstm_batch
        self.lstm_dropout = lstm_dropout
        # conv parameters
        self.conv1_size = conv1_size
        self.conv2_size = conv2_size
        self.conv3_size = conv3_size
        self.conv1_kernel = conv1_kernel
        self.conv2_kernel = conv2_kernel
        self.conv3_kernel = conv3_kernel
        # fc parameters
        self.fc_dropout = fc_dropout

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_num_layers, batch_first=self.lstm_batch)

        # arg 3 can be an int! ignore the tuple complaint by lint
        self.conv1 = nn.Conv1d(self.input_size, self.conv1_size, self.conv1_kernel)
        self.conv2 = nn.Conv1d(self.conv1_size, self.conv2_size, self.conv2_kernel)
        self.conv3 = nn.Conv1d(self.conv3_size, self.conv3_size, self.conv3_kernel)

        self.fc = nn.Linear(self.conv3_size + self.lstm_hidden_size, self.output_classn)

        self.bn1 = nn.BatchNorm1d(self.conv1_size)
        self.bn2 = nn.BatchNorm1d(self.conv2_size)
        self.bn3 = nn.BatchNorm1d(self.conv3_size)

        self.se1 = SELayer(self.conv1_size)
        self.se2 = SELayer(self.conv2_size)

        self.drop_lstm = nn.Dropout(self.lstm_dropout)
        #self.drop_fc = nn.Dropout(self.fc_dropout)

    def forward(self, x, batch_tlength):
        """
        X should be Batch X time sample x EEG channel
        (wouldn't batch size be determined by other pytorch code?)
        :param x:
        :param batch_tlength:
        :return:
        """
        # x1 = lstm pathway
        # x2 = conv1d pathway

        # packing sequence and feeding through lstm (dimension shuffle?)
        # pre packing x1 dimension would be Batch x Time x EEGchan
        # after packing x1 dimension would be (Batch x Time) x EEGchan
        x1 = nn.utils.rnn.pack_padded_sequence(x, batch_tlength, batch_first=True, enforce_sorted=False)
        x1, (h_state, c_state) = self.lstm(x1)
        x1, seq_len = nn.utils.rnn.pad_packed_sequence(x1, batch_first=True)
        # once it is padded again it should be returned to the original dimension
        # (since the batch_first argument is the same in both calls)

        # reference code only selects the final time dimension sample... why??
        # x1 = x1[:,-1,:]

        # weights are initialized with kaming_uniform() by default in pytorch, no need to
        # specify explicit initialization
        x2 = x.transpose(2, 1)
        x2 = nnF.relu(self.bn1(self.conv1(x2)))
        x2 = self.se1(x2)
        x2 = nnF.relu(self.bn1(self.conv2(x2)))
        x2 = self.se2(x2)
        x2 = nnF.relu(self.bn1(self.conv3(x2)))
        # use global average pooling: check for dimension requirements
        x2 = nnF.adaptive_avg_pool1d(x2)
        # x2 = torch.mean(x2, 2)

        # first dimension
        x_combined = torch.cat((x1, x2), dim=1)
        x_out = self.fc(x_combined)
        x_out = nnF.log_softmax(x_out, dim=1)

        return x_out

class MLSTM_FCN(nn.Module):
    def __init__(self, input_size, max_sequence_size, output_classn,
                 lstm_hidden_size=128, lstm_num_layers=1, lstm_batch=True,
                 conv1_size=128, conv2_size=256, conv3_size=128,
                 conv1_kernel=8, conv2_kernel=5, conv3_kernel=3,
                 lstm_dropout=0.8, conv_dropout=0.3):
        super().__init__()

        self.input_size = input_size
        self.max_sequence_size = max_sequence_size
        self.output_classn = output_classn
        # lstm parameters
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_batch = lstm_batch
        self.lstm_dropout = lstm_dropout
        # conv parameters
        self.conv1_size = conv1_size
        self.conv2_size = conv2_size
        self.conv3_size = conv3_size
        self.conv1_kernel = conv1_kernel
        self.conv2_kernel = conv2_kernel
        self.conv3_kernel = conv3_kernel
        # fc parameters
        self.conv_dropout = conv_dropout

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_num_layers, batch_first=self.lstm_batch)

        # arg 3 can be an int! ignore the tuple complaint by lint
        self.conv1 = nn.Conv1d(self.input_size, self.conv1_size, self.conv1_kernel)
        self.conv2 = nn.Conv1d(self.conv1_size, self.conv2_size, self.conv2_kernel)
        self.conv3 = nn.Conv1d(self.conv3_size, self.conv3_size, self.conv3_kernel)

        self.fc = nn.Linear(self.conv3_size+self.lstm_hidden_size, self.output_classn)

        self.bn1 = nn.BatchNorm1d(self.conv1_size)
        self.bn2 = nn.BatchNorm1d(self.conv2_size)
        self.bn3 = nn.BatchNorm1d(self.conv3_size)

        self.se1 = SELayer(self.conv1_size)
        self.se2 = SELayer(self.conv2_size)

        self.drop_lstm = nn.Dropout(self.lstm_dropout)
        self.drop_conv = nn.Dropout(self.conv_dropout)


    def forward(self, x, batch_tlength):
        """
        X should be Batch X time sample x EEG channel
        (wouldn't batch size be determined by other pytorch code?)
        :param x:
        :param batch_tlength:
        :return:
        """
        #x1 = lstm pathway
        #x2 = conv1d pathway

        # packing sequence and feeding through lstm (dimension shuffle?)
        # pre packing x1 dimension would be Batch x Time x EEGchan
        # after packing x1 dimension would be (Batch x Time) x EEGchan
        x1 = nn.utils.rnn.pack_padded_sequence(x, batch_tlength, batch_first=True, enforce_sorted=False)
        x1, (h_state, c_state) = self.lstm(x1)
        x1, seq_len = nn.utils.rnn.pad_packed_sequence(x1, batch_first=True)
        # once it is padded again it should be returned to the original dimension
        # (since the batch_first argument is the same in both calls)

        # reference code only selects the final time dimension sample... why??
        # x1 = x1[:,-1,:]


        x2 = x.transpose(2,1)
        x2 = self.drop_conv(nnF.relu(self.bn1(self.conv1(x2))))
        x2 = self.se1(x2)
        x2 = self.drop_conv(nnF.relu(self.bn1(self.conv2(x2))))
        x2 = self.se2(x2)
        x2 = self.drop_conv(nnF.relu(self.bn1(self.conv3(x2))))
        x2 = torch.mean(x2, 2)

        # first dimension
        x_combined = torch.cat((x1, x2), dim=1)
        x_out = self.fc(x_combined)
        x_out = nnF.log_softmax(x_out, dim=1)



        return