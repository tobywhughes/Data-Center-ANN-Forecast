﻿# Data-Center-ANN-Forecast

Instructions to Run

1. Create file called pcap_list.txt

2. In this file, list line by line all pcap files you wish to parse

Ex:

C:\file1.pcap

C:\file2.pcap

3. Run pcap_parser.py

4. Grab subset from ./cache/series_cache and place in folder called ./subsets

5. Currently, each keras implementation parses 4 different subsets, although this can easily be reconfigured. For easiest way to run, parse 4 seperate subsets with pcap parsers and name them using the names found in the read_subset_series functions in the models.

6. Run the models and wait.
