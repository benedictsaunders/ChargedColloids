import numpy as np

DE = {
 1.424: 0.897,
 2.466: 0.188,
 2.848: -0.066,
 3.767: 0.068,
 4.272: 0.033,
 4.933: 0.024,
 5.134: -0.015,
 5.696: 0.005,
 6.207: 0.001,
 6.525: -0.009,
 7.12: -0.025,
 7.399: 0.01,
 7.535: 0.005,
 7.928: -0.005,
 8.544: -0.009,
 8.662: -0.006,
 8.893: -0.011,
 9.338: -0.006,
 9.866: 0.005,
 9.968: 0.009,
 10.268: 0.0,
 10.751: -0.003,
 11.122: -0.008,
 11.302: -0.005,
 11.392: -0.004,
 11.656: -0.0,
 12.166: 0.003,
 12.332: 0.005,
 12.414: 0.004,
 12.656: 0.003,
 12.816: 0.002,
 13.051: 0.001,
 13.584: 0.0,
 13.732: 0.0,
 14.024: 0.001,
 14.24: 0.001,
 14.452: 0.001,
 14.798: 0.001,
 14.867: 0.001,
 15.002: 0.001,
 15.07: 0.001,
 15.403: 0.0,
 15.664: 0.0,
 15.857: 0.0,
 16.047: 0.0,
 16.173: 0.0,
 16.422: 0.0,
 16.788: 0.0
 }

class GrAMPLE:

    def __init__(self):
        x = np.array(list(DE.keys()))
        y = np.array(list(DE.values()))
        self.fit = np.polyfit(x, y, 7) # Polynomial fit of degree 7 # yeah yeah I know it's overkill, but its the best fit without going to other functions.
        # print("GrAMPLE initialized. It's called GrAMPLE because it is a Great Radical Approach to Modelling Practical Lukas' Epiphanies.")
        # print("Not because it's a portmanteau of graphene and SAMPLE.")

    def get_energy(self, d):
        return np.polyval(self.fit, d)