import argparse

class VBOTParser:
    def __init__(self):
        # create a parser
        self.parser = argparse.ArgumentParser(prog='exp_mot_lc', 
                                            description='Experiment VBOT Lane Change case',
                                            epilog='Try it !!')
        # add argument info 
        self.add_arguments()

        # parse and save the Namespace object
        self.args = self.parser.parse_args()


    def add_arguments(self):
        # add the arguments
        self.parser.add_argument('--norun',
                            default=False,
                            action='store_true'
                            )
        self.parser.add_argument('--plot',
                            default=False,
                            action='store_true'
                            )
        self.parser.add_argument('--batch',
                            default=False,
                            action='store_true'
                            )
        self.parser.add_argument('--k1',
                            action='store',
                            metavar='K_1 gain affects a_lat more (float)',
                            type=float,
                            )
        self.parser.add_argument('--k2',
                            action='store',
                            metavar='K_2 gain affects a_long more (float)',
                            type=float,
                            )
        self.parser.add_argument('--kw',
                            action='store',
                            metavar='K_W gain (float)',
                            type=float,
                            )
        self.parser.add_argument('--tf',
                            action='store',
                            metavar='FINAL_TIME (float)',
                            type=float,
                            required=False
                            )




'''
Eg.


p1_parser.add_argument('msg',
                       metavar='MESSAGE',
                       type=str,
                       help='Prints a message')

p1_parser.add_argument('--input',
                        action='store',
                        metavar='STRING_INPUT',
                        type=str,
                        required=True)

p1_parser.add_argument('--output',
                        action='store',
                        type=str)
p1_parser.add_argument('-n',
                        '--number',
                        action='store',
                        metavar='RATING 0-9',
                        choices=range(1,10),
                        type=int,
                        required=True)


There are several actions that are already defined and ready to be used. Let’s analyze them in detail:

'store' 
stores the input value to the Namespace object. (This is the default action.)

'store_const' 
stores a constant value when the corresponding optional arguments are specified.

'store_true' 
stores the Boolean value True when the corresponding optional argument is specified and stores a False elsewhere.

'store_false' 
stores the Boolean value False when the corresponding optional argument is specified and stores True elsewhere.

'append' 
stores a list, appending a value to the list each time the option is provided.

'append_const' 
stores a list appending a constant value to the list each time the option is provided.

'count' 
stores an int that is equal to the times the option has been provided.

'help' 
shows a help text and exits.

'version' 
shows the version of the program and exits.
'''


