#!/usr/bin/env python3

#--------------------------------------------------------------------------------
# Console client for AbletonOSC.
# Takes OSC commands and parameters, and prints the return value.
#--------------------------------------------------------------------------------

import re
import argparse
import readline

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

def main(args):
    client = AbletonOSCClient(args.hostname, args.port)
    if args.verbose:
        client.verbose = True
    client.send_message("/live/reload")

    readline.parse_and_bind('tab: complete')
    print("AbletonOSC command console")
    print("Usage: /live/osc/command [params]")

    while True:
        try:
            command_str = input(">>> ")
        except EOFError:
            print()
            break

        if not re.search("\\w", command_str):
            #--------------------------------------------------------------------------------
            # Command is empty
            #--------------------------------------------------------------------------------
            continue
        if not re.search("^/", command_str):
            #--------------------------------------------------------------------------------
            # Command is invalid
            #--------------------------------------------------------------------------------
            print("OSC address must begin with a slash (/)")
            continue

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
    parser = argparse.ArgumentParser(description="Console client for AbletonOSC. Takes OSC commands and parameters, and prints the return value.")
    parser.add_argument("--hostname", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default=11000)
    parser.add_argument("--verbose", "-v", action="store_true", help="verbose mode: prints all OSC messages")
    args = parser.parse_args()
    main(args)
