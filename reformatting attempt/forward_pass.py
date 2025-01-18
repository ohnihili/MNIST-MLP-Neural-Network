import numpy as np


# defines forward pass/propogation
def forward(self):
    # input to h1
        # finds the pre-activation values of each h1 node
    self.h1_pre = np.dot(self.w_i_h1,self.images.T) + self.b_i_h1
        # uses act-function to properly set each h1 node
    self.h1_a = self.sigmoid(self.h1_pre)

    # h1 to h2
    self.h2_pre = np.dot(self.w_h1_h2,self.h1_a) + self.b_h1_h2
    self.h2_a = self.sigmoid(self.h2_pre)

    # h2 to output
    self.out_pre = np.dot(self.w_h2_o, self.h2_a) + self.b_h2_o
    self.predictions = self.softmax(self.out_pre)
    
    # return the normalized predictions from the output layer
    return self.predictions