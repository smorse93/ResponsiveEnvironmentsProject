#!/usr/bin/env python3

#--------------------------------------------------------------------------------
# Console client for AbletonOSC.
# Takes OSC commands and parameters, and prints the return value.
#--------------------------------------------------------------------------------

import re
import argparse
import readline
import random

from client import AbletonOSCClient

class LiveAPICompleter:
    def __init__(self, commands):
        self.commands = commands

    def complete(self, text, state):
        results =  [x for x in self.commands if x.startswith(text)] + [None]
        return results[state]

words = ["horse", "hogan", "horrific"]
completer = LiveAPICompleter(words)
readline.set_completer(completer.complete)
parser = argparse.ArgumentParser(description="Console client for AbletonOSC. Takes OSC commands and parameters, and prints the return value.")
parser.add_argument("--hostname", type=str, default="127.0.0.1")
parser.add_argument("--port", type=str, default=11000)
parser.add_argument("--verbose", "-v", action="store_true", help="verbose mode: prints all OSC messages")
args = parser.parse_args()
client = AbletonOSCClient(args.hostname, args.port)


# This function isn't currently being used, but don't get rid of it

def main():        
    if args.verbose:
        client.verbose = True
    client.send_message("/live/reload")

    readline.parse_and_bind('tab: complete')
    print("AbletonOSC command console")
    print("Usage: /live/osc/command [params]")
    
def getTempo(command):
    
    bpm = client.query("/live/song/get/tempo")
    return bpm[0]

def doSomething(command):
    
    # This program translates a command from main.py to something Ableton can understand
    
    client.send_message("/live/reload")

    print(command)
    
    command_str = command
    command, *params_str = command_str.split(" ")
    params = []
    for part in params_str:
        try:
            part = int(part)
        except ValueError:
            try:
                part = float(part)
            except ValueError:
                pass
        params.append(part)
    try:
        print(client.query(command, params))
    except RuntimeError:
        pass


if __name__ == "__main__":
    main()
