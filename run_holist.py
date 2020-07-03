# This script contains parts of code from deephol main source files.

import argparse
import os
import random
from multiprocessing import Pool

import tensorflow as tf
from google.protobuf import text_format

from protobuf_python import theorem_database_pb2


def translate_splits(splits):
  """Translate a comma separated list of splits in to python set.

  Args:
    splits: String with comma separated list of split specifications.

  Returns:
    Python set of theorem_database_pb2.Theorem.Split.
  """

  def translate(s):
    if s == 'testing':
      return theorem_database_pb2.Theorem.TESTING
    elif s == 'training':
      return theorem_database_pb2.Theorem.TRAINING
    elif s == 'validation':
      return theorem_database_pb2.Theorem.VALIDATION
    print('Unknown split specification: %s' % s)

  if splits == 'all':
    return translate_splits('training,testing,validation')
  return {translate(s) for s in splits.split(',')}


def load_theorem_database_from_file(filename):
    """Load a theorem database from a text protobuf file."""
    theorem_database = theorem_database_pb2.TheoremDatabase()
    with tf.gfile.Open(filename) as f:
        text_format.MergeLines(f, theorem_database)
    print('Successfully read theorem database from %s (%d theorems).' % (filename, len(theorem_database.theorems)))
    return theorem_database


def is_thm_included(thm, splits, library_tags):
    """Decides whether the theorem is included in the selection.

    This function can be used for filtering for theorems belonging to
    the allowed splits and library tags.

    Args:
    thm: Theorem object to be decided for inclusion.
    splits: Of type List[proof_assistant_pb2.Theorem.Split], the list of
    training splits for which tasks should be generated for.
    library_tags: List of strings for the library tags to be processed. If
    empty, then all library tags are allowed.

    Returns:
    Boolean indicating whether the theorem is included in the selection.
    """
    return (thm.training_split in splits) and (not library_tags or (set(thm.library_tag) & library_tags))


def get_fingerprints(database, split, library):
    libraries = None
    if library is not None:
        libraries = {library}

    return [thm.fingerprint for thm in database.theorems
            if is_thm_included(thm, split, libraries)]


def reset(num_instances, exp_name, holist_data_mount):
    os.system("sudo rm -rf %s/logs/%s" % (holist_data_mount, exp_name))
    os.system("sudo rm -rf %s/prooflogs/%s" % (holist_data_mount, exp_name))

    os.system("mkdir %s/logs/%s" % (holist_data_mount, exp_name))
    os.system("mkdir %s/prooflogs/%s" % (holist_data_mount, exp_name))

    for i in range(num_instances):
        os.system("docker stop holist%d && docker rm holist%d && docker network rm holist_net%d" % (i, i, i))


def run_HOLights(num_instances):
    for i in range(num_instances):
        os.system("docker network create holist_net%d" % (i,))
        os.system("docker run -d --network=holist_net%d --name=holist%d gcr.io/deepmath/hol-light" % (i, i))


def compute_embeddings(holist_data_mount):
    # Last parameter will override any set of prover tasks,
    # so we will only compute and save embeddings (if they are not computed already).
    command = 'docker run -it --network=holist_net0 -v %s:/data gcr.io/deepmath/deephol ' \
              '--prover_options=/data/configuration/prover_options.textpb ' \
              '--output="/data/prooflogs/prooflog_empty.textpbs" --proof_assistant_server_address=holist0:2000 ' \
              '--tasks_by_fingerprint="42"' % (holist_data_mount,)

    print("WARNING: Please make sure you set up correct theorem database and model checkpoint in prover options!")
    print("INFO: Computing embeddings!")
    os.system(command)


def prove(fingerprints, num_instances, holist_data_mount, exp_name):
    print("INFO: Proving! You will find logs for each instance in logs directory.")

    batches = [[] for _ in range(num_instances)]
    for i, fingerprint in enumerate(fingerprints):
        batches[i % num_instances].append(fingerprint)

    print("Sizes of fingerprints's batches:")
    for batch in batches:
        print(len(batch), end=" ")

    print("\n\n\n")

    # Watch out! Concurrent part!
    def get_HOList_command(num, theorems, data_mount):
        theorems_str = ""
        for thm in theorems:
            theorems_str += str(thm) + ","
        theorems_str = theorems_str[:-1]

        command = 'docker run -it --network=holist_net%d -v %s:/data gcr.io/deepmath/deephol ' \
                  '--prover_options=/data/configuration/prover_options.textpb ' \
                  '--output="/data/prooflogs/%s/prooflog%d.textpbs" --proof_assistant_server_address=holist%d:2000 ' \
                  '--tasks_by_fingerprint="%s" > %s/logs/%s/holight_log%d.out' \
                  % (num, data_mount, exp_name, num, num, theorems_str, data_mount, exp_name, num)

        return command

    with Pool(processes=num_instances) as pool:
        commands = [get_HOList_command(i, batches[i], holist_data_mount) for i in range(num_instances)]
        pool.map(os.system, commands)


def write_summary(num_instances, holist_data_mount, exp_name):
    print("INFO: Aggregating summaries from prooflogs\n\n\n")

    prooflogs = ""
    for i in range(num_instances):
        prooflogs += "/data/prooflogs/%s/prooflog%d.textpbs," % (exp_name, i)
    prooflogs = prooflogs[:-1]

    command = 'docker run -it --network=holist_net0 -v %s:/data gcr.io/deepmath/deephol ' \
              '--prover_options=/data/configuration/prover_options.textpb ' \
              '--output="%s" --proof_assistant_server_address=holist0:2000 ' \
              '--tasks_by_fingerprint="42"' % (holist_data_mount, prooflogs)

    os.system(command)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--database_path", help="Path to database with theorems.")
    parser.add_argument("--split", default="validation", type=str, choices=['validation', 'testing'],
                        help="Split of theorem database that we want to prove. Should be equal 'validation' or 'testing'.")
    parser.add_argument("--library", default=None, type=str,
                        help="Specify a library of theorems we want to prove. "
                             "If None we will prove all (with respect to chosen split).")
    parser.add_argument("--num_instances", default=16, type=int,
                        help="Number of HOList/DeepHOL instances to start."
                             "Theorems to prove will be randomly and evenly split amongst them.")
    parser.add_argument("--holist_data_mount", help="Path to directory with configuration and checkpoints subdirectories."
                                                    "This path will be mounted in HOList docker under /data.")
    parser.add_argument("--only_summary", type=bool, default=False,
                        help="Specify whether you want to calculate summary of proofs done from prooflogs or prove selected theorems.")
    parser.add_argument("--exp_name", type=str, help="Name of the experiment.")

    args = parser.parse_args()

    if args.exp_name is None:
        raise ValueError("You must provide experiment name!")

    if args.database_path is None:
        raise ValueError("You must provide theorem database path!")

    if args.num_instances <= 0:
        raise ValueError("Number of HOList/DeepHOL instances shoud be positive!")

    if not args.holist_data_mount:
        raise ValueError("You must specify HOList /data mount path!")

    database = load_theorem_database_from_file(args.database_path)
    fingerprints = get_fingerprints(database, translate_splits(args.split), args.library)

    print("Found %d theorems to prove." % (len(fingerprints),))

    reset(args.num_instances, args.exp_name, args.holist_data_mount)
    run_HOLights(args.num_instances)

    if not args.only_summary:
        compute_embeddings(args.holist_data_mount)
        prove(fingerprints, args.num_instances, args.holist_data_mount, args.exp_name)

    write_summary(args.num_instances, args.holist_data_mount, args.exp_name)


if __name__ == "__main__":
    main()
