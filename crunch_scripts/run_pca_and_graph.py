#!/usr/bin/env python
"""
Purpose: Load numpy files and run PCA

Does not use parallelization since it analyzes all the population at once

Inputs:
  Required
    'pythonic-tiling-callset-files' : Collection containing callsets encoded with pythonic tilings.
                                      (Such a collection is output by concatenate-pythonic-tilings)
                                      Will contain some directory substructure of the following form:
        callset-name/
            quality_phase0.npy
            quality_phase1.npy
            phase0.npy
            phase1.npy
        callset-name/
        ...
    'pythonic-tiling-info' : Collection output by make-pythonic-tiling-info
        path_integers.npy
        path_lengths.npy
    'png-name' : name to prepend to output files (text)
    'percent-to-analyze' : percent to analyze (number)
    'in-depth-component-analysis': Text interpreted as boolean. if true, backsolves the components to find out which tile positions are weighted heavily in the PCA.
        Asserts that percent-to-analyze is 100. Takes more memory and time
    'quality' : Text, interpreted as boolean. If true, uses quality_phase files. If false, uses phase files
  Optional:
    'callset-phenotypes' : Collection with files containing phenotypic information
    'callset-name-regex' : String interpreted as regex, used to pull names out of the numpy files and callset-phenotype files
    'num-retries' : integer for number of retries to use when saving. Defaults to 3
    'accepted-paths' : Json-formatted text (list of lists) used to specify which paths should be converted to library format. By default, all paths in 'input' are
                       converted. Path bounds are 0-indexed and the upper path is exclusive. If a path is specified here, but is not in 'input', it will not be
                       written.
                       Ex: [ ["000","002"], ["00f","014"]] will result in paths 000, 001, 00f, 010, 011, 012, and 013 being converted to library format.

Outputs:
    [png-name]_PCA.png
    time_log.txt
    [png-name]_PCA_first_component.png [if in-depth-component-analysis is true (and therefore percent-to-analyze == 100)]
    [png-name]_PCA_second_component.png [if in-depth-component-analysis is true (and therefore percent-to-analyze == 100)]
"""
import helping_functions as fns # must be imported first to ensure matplotlib visualization works

import arvados      # Import the Arvados sdk module
import re           # used for error checking
import json
import os
import time
import numpy as np

def get_bool(param):
    if param.lower() == 'true' or param.lower() == 't':
        return True
    elif param.lower() == 'false' or param.lower() == 'f':
        return False
    raise Exception("%s cannot be parsed as boolean. Please use true/false." % (param))

# Read constants
########################################################################################################################
#Get name to save png file under
PNG_NAME = arvados.getjobparam('png-name')
#Get whether we should use quality or regular numpy files
QUALITY = get_bool(arvados.getjobparam('quality'))
#Get percentage of the tiles we should use in PCA
PERCENT_TO_RETRIEVE = float(arvados.getjobparam('percent-to-analyze'))
#Get if we should only load certain parts of the genome
ACCEPTED_PATHS = []
input_accepted_paths = arvados.getjobparam('accepted-paths')
if input_accepted_paths != None:
    try:
        ACCEPTED_PATHS = json.loads(input_accepted_paths)
    except ValueError:
        raise Exception("Unable to read 'accepted-paths' input as json input: %s" % (input_accepted_paths))
    for lower_path, upper_path in ACCEPTED_PATHS:
        assert re.match('^[0-9a-f]+$', lower_path) != None, \
            "'accepted-paths' input contains an incorrect path hex string (%s) (it does not match '^[0-9a-f]+$')" % (lower_path)
        assert re.match('^[0-9a-f]+$', upper_path) != None, \
            "'accepted-paths' input contains an incorrect path hex string (%s) (it does not match '^[0-9a-f]+$')" % (upper_path)
        assert int(lower_path,16) < int(upper_path,16), "for each pair in 'accepted-paths', expects the first path to be less than the second"
#Get if we should do in-depth-component analysis
IN_DEPTH_COMPONENT_ANALYSIS = get_bool(arvados.getjobparam('in-depth-component-analysis'))
if IN_DEPTH_COMPONENT_ANALYSIS:
    assert PERCENT_TO_RETRIEVE == 100.0, "if 'in-depth-component-analysis' is true, 'percent-to-analyze' must be equal to 100"
    assert ACCEPTED_PATHS == [], "if 'in-depth-component-analysis' is true, 'accepted-paths' must be left blank or be equal to []"
#Get how many times we should try to retrieve collections
NUM_RETRIES = arvados.getjobparam('num-retries') or 3
NUM_RETRIES = int(NUM_RETRIES)
assert NUM_RETRIES > 0, "'num-retries' must be strictly positive"
#Get if we should have a different callset regex
CALLSET_NAME_REGEX = "(hu[0-9A-F]+|HG[0-9]+|NA[0-9]+|GS[0-9]+)"
if arvados.getjobparam('callset-name-regex') != None:
    CALLSET_NAME_REGEX = arvados.getjobparam('callset-name-regex')
########################################################################################################################
#Set-up collection and logging file to write out to
out = arvados.collection.Collection(num_retries=NUM_RETRIES)
time_logging_fh = out.open('time_log.txt', 'w')
info_logging_fh = out.open('info_log.txt', 'w')
########################################################################################################################
#Get path lengths and path integers
cr = arvados.CollectionReader(arvados.getjobparam('pythonic-tiling-info'), num_retries=NUM_RETRIES)
t0 = time.time()
with cr.open("path_integers.npy", 'r') as f:
    path_integers = np.load(f)
t1 = time.time()
with cr.open("path_lengths.npy", 'r') as f:
    path_lengths = np.load(f)
t2 = time.time()
time_logging_fh.write('Loading path integers took %fs\n' %(t1-t0))
time_logging_fh.write('Loading path lengths took %fs\n' %(t2-t1))

########################################################################################################################
#Get callset_collection_reader
t0 = time.time()
callset_collection_reader = arvados.CollectionReader(arvados.getjobparam('pythonic-tiling-callset-files'))
callset_collection_reader.normalize()
t1 = time.time()
time_logging_fh.write("Opening and normalizing 'pythonic-tiling-callset-files' took %fs\n" % (t1-t0))
NUM_PHASES_TMP = 0
NUM_CALLSETS = 0
t0 = time.time()
for s in callset_collection_reader.all_streams():
    if s.name() != '.':
        if re.search(CALLSET_NAME_REGEX, s.name()) == None:
            print CALLSET_NAME_REGEX, s.name(), "did not match search?"
        for f in s.all_files():
            if f.name().startswith('quality_') and QUALITY:
                NUM_PHASES_TMP += 1
            elif not QUALITY:
                NUM_PHASES_TMP += 1
        NUM_CALLSETS += 1
t1 = time.time()
NUM_PHASES = NUM_PHASES_TMP/NUM_CALLSETS
assert float(NUM_PHASES) == NUM_PHASES_TMP/float(NUM_CALLSETS), "Unequal number of phases per callset"
time_logging_fh.write("Cursory reading of 'pythonic-tiling-callset-files' took %fs\n" % (t1-t0))

#Get callset phenotype files
#Unable to use collections due to csv/json read functions
t0 = time.time()
if arvados.getjobparam('callset-phenotypes') == None:
    phenotype_file_paths = []
else:
    phenotype_path = arvados.get_job_param_mount('callset-phenotypes')
    for root, dirs, files in os.walk(phenotype_path):
        assert len(dirs) == 0, "Expects 'callset-phenotypes' to be a flat directory"
        phenotype_file_paths = [os.path.join(root, f) for f in files]
t1 = time.time()
time_logging_fh.write("Getting job param mount (and file paths) of 'callset-phenotypes' took %fs\n" % (t1-t0))
########################################################################################################################
population, subjects, callset_names, size = fns.get_population(
    ACCEPTED_PATHS,
    path_integers,
    path_lengths,
    NUM_CALLSETS,
    NUM_PHASES,
    phenotype_file_paths,
    callset_collection_reader,
    CALLSET_NAME_REGEX,
    time_logging_fh,
    QUALITY
)

png_file_handle1 = out.open(PNG_NAME+'.png', 'w')
png_file_handle2 = out.open('intense_colors_'+PNG_NAME+'.png', 'w')
if IN_DEPTH_COMPONENT_ANALYSIS:
    # Perform PCA on all the population and get results you can backsolve from (takes more memory and time)
    well_seq_locations, number_of_columns_per_spread, pca = fns.PCA_random_portion(
        population,
        subjects,
        PERCENT_TO_RETRIEVE,
        callset_names,
        time_logging_fh,
        space_efficient=False,
        png_fh=[png_file_handle1, png_file_handle2]
    )
    with out.open(PNG_NAME+"_first_component.png", 'w') as second_png_handle:
        #Draw the components PCA used to project data onto the first principal component
        fns.draw_pca_weights(
            well_seq_locations,
            pca.components_[0],
            number_of_columns_per_spread,
            size,
            second_png_handle
        )
    with out.open(PNG_NAME+"_second_component.png", 'w') as second_png_handle:
        #Draw the components PCA used to project data onto the second principal component
        fns.draw_pca_weights(
            well_seq_locations,
            pca.components_[1],
            number_of_columns_per_spread,
            size,
            second_png_handle,
            ylabel="Weight used to project onto the second principal component"
        )
else:
    # Perform PCA on all the population: just get the PCA results
    colors = fns.make_ethnicity_colors(subjects, callset_names)
    #Filter population
    time_logging_fh.write("Original population size is" + str(population.shape)+ "\n")
    if PERCENT_TO_RETRIEVE < 100:
        print "Filtering population, starting shape (%s), num MB (%f), max variant value (%i)" % (population.shape, population.nbytes/1000000., np.amax(population))
        t1 = time.time()
        filter_pick, population = fns.random_filter_population(population, PERCENT_TO_RETRIEVE)
        del filter_pick
        t2 = time.time()
        time_logging_fh.write("Randomly filtering population took %f seconds\n" % (t2-t1))
        time_logging_fh.write("Randomly filtered population size is"+ str(population.shape) + "\n")
    else:
        t2 = time.time()
        t1 = t2
    print "Reducing population, starting shape (%s), num MB (%f), max variant value (%i)" % (population.shape, population.nbytes/1000000., np.amax(population))
    population, where_well_seq, altered = fns.reduce_to_minimum(population)
    del altered
    del where_well_seq
    t3 = time.time()
    time_logging_fh.write("Altering population took %f seconds\n" % (t3-t2))
    time_logging_fh.write("Altered population size is" + str(population.shape)+"\n")
    #spread randomly chosen population
    print "Spreading population, starting shape (%s), num MB (%f), max variant value (%i)" % (population.shape, population.nbytes/1000000., np.amax(population))
    population, matching = fns.spread_population(population, True)
    del matching
    t4 = time.time()
    time_logging_fh.write("Spreading population took %f seconds\n" % (t4-t3))
    time_logging_fh.write("Spread population size is" + str(population.shape)+"\n")
    #run 2-component PCA on spread population
    print "Running PCA, starting shape (%s), num MB (%f), max variant value (%i)" % (population.shape, population.nbytes/1000000., np.amax(population))
    pca, population = fns.run_PCA(2, population)
    del pca
    t5 = time.time()
    time_logging_fh.write("2-component PCA took %f seconds\n" % (t5-t4))
    fns.two_d_plot(population, colors, png_fh=png_file_handle1)
    intense_colors = fns.make_colors(subjects, 'ethnicity', callset_names, fns.pop_human_readable)
    time_logging_fh.write("Intense color mapping: %s\n" % (intense_colors))
    fns.two_d_plot(population, intense_colors, png_fh=png_file_handle2)
    info_logging_fh.write('projected_population=%s\n' % (population.tolist()))
    info_logging_fh.write('intense_colors=%s\n' % (intense_colors))
    info_logging_fh.write('colors=%s\n' % (colors))
    info_logging_fh.write('subjects=%s\n' % (subjects))
    info_logging_fh.write('callset_names=%s\n' % (callset_names))


time_logging_fh.close()
png_file_handle1.close()
png_file_handle2.close()
# Commit the output to keep
task_output = out.save_new(create_collection_record=False)
arvados.current_task().set_output(task_output)

###########################################################################################################################################
