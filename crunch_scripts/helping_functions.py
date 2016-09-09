import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os
import re
import csv
import time

# Copied from http://www.1000genomes.org/category/phenotype
pop_to_super_pop = {
  "CHB":"EAS",
  "JPT":"EAS",
  "CHS":"EAS",
  "CDX":"EAS",
  "KHV":"EAS",
  "CEU":"EUR",
  "TSI":"EUR",
  "FIN":"EUR",
  "GBR":"EUR",
  "IBS":"EUR",
  "YRI":"AFR",
  "LWK":"AFR",
  "MSI":"AFR",#added
  "GWD":"AFR",
  "MSL":"AFR",
  "ESN":"AFR",
  "ASW":"AFR",
  "ACB":"AFR",
  "MXL":"AMR",
  "PUR":"AMR",
  "CLM":"AMR",
  "PEL":"AMR",
  "GIH":"SAS",
  "PJL":"SAS",
  "BEB":"SAS",
  "STU":"SAS",
  "ITU":"SAS",
  "asian-pgp":"EAS",
  "white-pgp":"EUR",
  "ethiopian-pgp":"AFR",
  "columbian-pgp":"AMR"
}
pop_human_readable = {
  "CHB":["Han Chinese in Bejing, China",'yellowgreen'], #None in 433
  "JPT":["Japanese in Tokyo, Japan",'mediumspringgreen'], #None in 433
  "CHS":["Southern Han chinese",'forestgreen'], #93 in 433
  "CDX":["Chinese Dai in Xishuangbanna, China",'mediumseagreen'], #None in 433
  "KHV":["Kinh in Ho Chi Minh City, Vietnam", 'lawngreen'], #6 in 433
  "asian-pgp":["PGP participants of Asian ancestry", 'olive'],

  "CEU":["Utah Residents (CEPH) with Northern and Western European Ancestry",'darkviolet'], #96 in 433
  "TSI":["Toscani in Italia",'darkslateblue'], #None in 433
  "FIN":["Finnish in Finland",'blueviolet'], #None in 433
  "GBR":["British in England and Scotland",'darkmagenta'], #None in 433
  "IBS":["Iberian Population in Spain",'indigo'], #None in 433
  "white-pgp":["PGP participants of European ancestry",'magenta'],

  "YRI":["Yoruba in Ibadan, Nigeria",'maroon'], #80 in 433
  "LWK":["Luhya in Webuye, Kenya",'sandybrown'], #11 in 433
  "MSI":["Maasai from Kenya", 'peru'],#None in 433, added!
  "GWD":["Gambian in Western Divisions in the Gambia",'sienna'], #None in 433
  "MSL":["Mende in Sierra Leone",'goldenrod'], #None in 433
  "ESN":["Esan in Nigeria",'tan'], #None in 433
  "ASW":["Americans of African Ancestry in SW USA",'brown'], #None in 433
  "ACB":["African Caribbeans in Barbados",'burlywood'], #None in 433
  "ethiopian-pgp":["PGP Ethiopian participant", 'black'],

  "MXL":["Mexican Ancestry from Los Angeles USA",'tomato'], #None in 433
  "PUR":["Puerto Ricans from Puerto Rico",'orangered'], #6 in 433
  "CLM":["Colombians from Medellin, Colombia",'coral'], #None in 433
  "PEL":["Peruvians from Lima, Peru",'palevioletred'], #94 in 433
  "columbian-pgp":["PGP Columbian participant", 'darkorange'],

  "GIH":["Gujarati Indian from Houston, Texas",'cyan'], #None in 433
  "PJL":["Punjabi from Lahore, Pakistan",'darkturquoise'], #47 in 433
  "BEB":["Bengali from Bangladesh", 'darkcyan'], #None in 433
  "STU":["Sri Lankan Tamil from the UK",'aquamarine'], #None in 433
  "ITU":["Indian Telugu from the UK",'lightseagreen'], #None in 433
}
super_pop_human_readable = {
  "EAS":["East Asian", 'g'],
  "EUR":["European", 'darkviolet'],
  "AFR":["African", 'brown'],
  "AMR":["Ad Mixed American", 'darkorange'],
  "SAS":["South Asian", 'c']
}

def get_subjects(subject_handle, subjects, mapping={'name':'Sample', 'Gender':'gender', 'Population':'ethnicity'}):
    """
    Extracts subject information from on csv file

    Returns subjects filled with pertinant information
        subjects is a dictionary keyed by huID.
    """
    survey_reader = csv.reader(subject_handle)
    new_mapping = {}
    name_index = None
    for i, row in enumerate(survey_reader):
        if i == 0: #We are reading the header
            assert mapping['name'] in row, "Unable to find the name-key (%s) in the header" % (mapping['name'])
            for key in mapping:
                if key != 'name':
                    csv_index = row.index(key)
                    new_mapping[csv_index] = mapping[key]
                else:
                    name_index = row.index(mapping[key])
        else:
            callset_name = row[name_index]
            if callset_name in subjects:
                for key in new_mapping:
                    value = row[key]
                    name_to_save_under = new_mapping[key]
                    if subjects[callset_name][name_to_save_under] == None:
                        subjects[callset_name][name_to_save_under] = value
                    else:
                        assert subjects[callset_name][name_to_save_under] == value, \
                            "Duplicate conflicting information %s is not %s for callset %s" % (subjects[callset_name][name_to_save_under], value, callset_name)
    return subjects

def shorten_callset_if_necessary(callset, accepted_path_indices, path_lengths, size):
    if accepted_path_indices == []:
        return callset
    accepted_callset = np.zeros((size,), dtype=np.int32)
    prev_index = 0
    for (start_path_index, end_path_index) in accepted_path_indices:
        start_index = path_lengths[start_path_index]
        end_index = path_lengths[end_path_index]
        piece_size = end_index - start_index
        accepted_callset[prev_index:prev_index+piece_size] = callset[start_index:end_index]
        prev_index += piece_size
    return accepted_callset

def get_population(accepted_paths, path_integers, path_lengths, num_callsets, num_phases, phenotype_file_paths, callset_collection_reader, callset_name_regex, logging_fh, quality):
    """
    Requires num_callsets and num_phases for preallocation of memory - speed performance

    Expects path_lengths to be a numpy array with a length of NUM_PATHS + 1.
        First entry should be 0. All entries should be greater than or
        equal to the others (since it indicates the number of tiles contained in
        all paths previous to its index). The last entry indicates the total
        number of tiles.
    Expects phenotype_file_paths to be a list of paths to csv files with callset information.
    Expects callset_collection_reader to be a _normalized_ CollectionReader

    Returns population, subjects
        population is a 2D numpy file of shape:
            (number_of_people_in_population, number_of_tiles_to_analyze)
            phases are concatonated together
        subjects is a dictionary keyed by callset name. Values are a dictionary:
            Keys are 'gender', 'ethnicity', and 'index'

    """
    accepted_path_indices = []
    size = 0
    for (start_path_string, end_path_string) in accepted_paths:
        try:
            start_path_int = int(start_path_string, 16)
            end_path_int = int(end_path_string, 16)
            #Want to add error checking before going public
            start_path_index = np.where(path_integers==start_path_int)[0][0]
            end_path_index = np.where(path_integers==end_path_int)[0][0]
            accepted_path_indices.append([start_path_index, end_path_index])
            size += path_lengths[end_path_index] - path_lengths[start_path_index]
        except ValueError:
            raise Exception("path_lengths, according to path_integers does not define paths (%s, %s) (from accepted_paths). Modify path_lengths input or accepted_paths input." % (start_path_string, end_path_string))
    if accepted_paths == None:
        size = path_lengths[-1]
    callset_names = []
    population = np.zeros((num_callsets, size*num_phases), dtype=np.int32)
    ## Fill population with tile identifiers
    t0 = time.time()
    callset_index= 0
    for s in callset_collection_reader.all_streams():
        if s.name() != '.':
            if re.search(callset_name_regex, s.name()) == None:
                print callset_name_regex, s.name(), "did not match search?"
            callset_names.append((s.name(), re.search(callset_name_regex, s.name()).group(0)))
            phase_index = 0
            for f in s.all_files():
                if f.name().startswith('quality_') and quality:
                    assert f.name() == 'quality_phase%i.npy' % (phase_index), \
                        "Expects 'callset-numpy-cgf-files' to be the output from concat-numpy files. %s is not 'quality_phase%i.npy'" % (f.name(), phase_index)
                    with callset_collection_reader.open(s.name()+'/'+f.name(), 'r') as f_handle:
                        callset = np.load(f_handle)
                        callset = shorten_callset_if_necessary(callset, accepted_path_indices, path_lengths, size)
                        population[callset_index][phase_index*size:(phase_index+1)*size] = callset
                    print s.name(), f.name(), "population number of MB: %f" % (population.nbytes/1000000.)
                    phase_index += 1
                elif not quality and not f.name().startswith('quality_'):
                    assert f.name() == 'phase%i.npy' % (phase_index), \
                        "Expects 'callset-numpy-cgf-files' to be the output from concat-numpy files. %s is not 'phase%i.npy'" % (f.name(), phase_index)
                    with callset_collection_reader.open(s.name()+'/'+f.name(), 'r') as f_handle:
                        callset = np.load(f_handle)
                        callset = shorten_callset_if_necessary(callset, accepted_path_indices, path_lengths, size)
                        population[callset_index][phase_index*size:(phase_index+1)*size] = callset
                    print s.name(), f.name(), "population number of MB: %f" % (population.nbytes/1000000.)
                    phase_index += 1
            callset_index += 1
    t1 = time.time()
    subjects = {callset_name:{'gender':None, 'ethnicity':None, 'index':i} for i, (dirname, callset_name) in enumerate(callset_names)}
    ## Fill subjects with phenotypic information if it is available
    #for s in phenotype_collection_reader.all_streams():
    #    for f in s.all_files():
    #        with phenotype_collection_reader.open(s.name()+'/'+f.name(), 'r') as f_handle:
    for path in phenotype_file_paths:
        with open(path, 'r') as f:
            subjects = get_subjects(f, subjects)

    t2 = time.time()
    logging_fh.write("Subject generation took %f seconds\n" % (t2-t1))
    logging_fh.write("Population generation takes %f seconds\n" % (t1-t0))
    return population, subjects, callset_names, size

def random_filter_population(population, percent):
    """
        percent is expected to be multiplied by 100:
        ie. 100 -> entire population
            2 -> returns 2% of the population
    """
    num_to_pick = int(population.shape[1]*0.01*percent)
    pick = np.random.choice(population.shape[1], num_to_pick, replace=False)
    retarray = np.zeros((population.shape[0], num_to_pick))
    for i, column in enumerate(pick):
        retarray[:,i] = population[:,column]
    return pick, retarray

def reduce_to_minimum(filtered_pop):
    ### Remove tiles (columns) that contain a -1, indicating it was not well sequenced
    not_seq_indicator = np.amin(filtered_pop, axis=0)
    sequenced_pick = np.greater_equal(not_seq_indicator, np.zeros(not_seq_indicator.shape))
    all_seq_pop = np.zeros((filtered_pop.shape[0], np.sum(sequenced_pick)))
    i = 0
    for column, picked in enumerate(sequenced_pick):
        if picked:
            all_seq_pop[:,i] = filtered_pop[:,column]
            i += 1
    ### spread_population runs in space O(max(tile_variant_int)). This means that
    ### the way population currently represents spanning tiles will make almost all
    ### machines run out of memory. So reduce those rows and indicate which ones they
    ### are so backsolving is possible
    variation_array = np.amax(all_seq_pop, axis=0)
    altered_pick = np.less_equal(variation_array, np.ones(variation_array.shape)*int('fff', 16))
    minimum_population = np.zeros(all_seq_pop.shape)
    for column, picked in enumerate(altered_pick):
        if picked:
            minimum_population[:,column] = all_seq_pop[:,column]
        else:
            unique_vals, ret_inverse = np.unique(all_seq_pop[:,column], return_inverse=True)
            minimum_population[:,column] = ret_inverse
    return minimum_population, sequenced_pick, altered_pick

def spread_population(minimum_population, space_efficient):
    """
    running spread_population without reduce_to_minimum is likely to create
        an out-of-memory error!
    space_efficient means the matrix takes less ram, but that backsolving
        which tile position is indicated by each row is not implemented yet
    Creates an indicator matrix indicating which variants
        are populated in the population. Will create a large return matrix
    """
    max_num_variation = np.amax(minimum_population)
    print "Maximum variant value in population:", max_num_variation
    spread_population = np.equal.outer(
        minimum_population,
        np.arange(max_num_variation)
    ).reshape(minimum_population.shape[0], -1).astype(int)
    if space_efficient:
        spread_population = spread_population[:, ~np.all(spread_population==0, axis=0)]
        return spread_population, None
    else:
        return spread_population, int(max_num_variation)

def two_d_plot(X, colors, ethnicity_to_colors=super_pop_human_readable, xlabel="Principal Component 1", ylabel="Principal Component 2", png_fh=None):
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=colors, alpha=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    rectangles = []
    rect_names = []
    for pop_abbr in ethnicity_to_colors:
        rectangles.append(plt.Rectangle((0, 0), 1, 1, fc=ethnicity_to_colors[pop_abbr][1]))
        rect_names.append(ethnicity_to_colors[pop_abbr][0])
    if len(rectangles) > 0:
        rectangles.append(plt.Rectangle((0, 0), 1, 1, fc='w'))
        rect_names.append('Unknown/Not Given')
        #plt.legend(tuple(rectangles), tuple(rect_names), loc="best")
    if png_fh != None:
        plt.savefig(png_fh)

def draw_pca_weights(well_seq, components, num_per_column, max_tile_int, png_fh, ylabel="Weight used to project onto the first principal component"):
    assert num_per_column != None, "Expects to have an integer number_of_columns_per_spread. Run PCA_random_portion with space_efficient=False"
    pca_x_values = []
    #Get indexes of well sequenced locations
    well_seq_locations = np.where(well_seq)[0]
    for i in range(len(components)):
        #convert into non-spread out index
        non_spread_x_i = i/num_per_column
        #append index associated with that component
        pca_x_values.append(well_seq_locations[non_spread_x_i])
    plt.figure()
    plt.scatter(pca_x_values, components)
    plt.xlabel('Tile position')
    plt.ylabel(ylabel)
    curr_axis = plt.axis()
    plt.axis([0,max_tile_int,curr_axis[2], curr_axis[3]])
    plt.savefig(png_fh)

def run_PCA(num_components, population):
    """
    Something that might be useful later:
        first_vector_by_component_size = sorted(enumerate(pca.components_[0]), key=lambda p: -abs(p[1]))
    """
    pca = PCA(n_components=num_components)
    X = pca.fit_transform(population)
    return pca, X

def make_colors(subjects, subjects_key, callset_names, key_to_colors):
    colors = []
    for callset_name in callset_names:
        val = subjects[callset_name[1]][subjects_key]
        if val == None:
            colors.append('w')
        else:
            colors.append(key_to_colors[val][1])
    return colors

def make_ethnicity_colors(subjects, callset_names):
    key_to_colors = {}
    for abbr in pop_to_super_pop:
        super_abbr = pop_to_super_pop[abbr]
        key_to_colors[abbr] = super_pop_human_readable[super_abbr]
    return make_colors(subjects, 'ethnicity', callset_names, key_to_colors)

def make_sex_colors(subjects, callset_names):
    key_to_colors = {
        'female':['female', 'm'],
        'male':['male', 'c'],
        '':['not known', 'g']
    }
    return make_colors(subjects, 'gender', callset_names, key_to_colors)

def PCA_random_portion(population, subjects, percent, callset_names, logging_fh, space_efficient=True, png_fh=None):
    colors = make_ethnicity_colors(subjects, callset_names)

    #Filter population
    logging_fh.write("Original population size is" + str(population.shape)+ "\n")
    if percent < 100:
        print "Filtering population, starting num MB: %f" % (population.nbytes/1000000.)
        t1 = time.time()
        filter_pick, population = random_filter_population(population, percent)
        del filter_pick
        t2 = time.time()
        logging_fh.write("Randomly filtering population took %f seconds\n" % (t2-t1))
        logging_fh.write("Randomly filtered population size is"+ str(population.shape) + "\n")
    else:
        t2 = time.time()
        t1 = t2
    print "Reducing population, starting num MB: %f" % (population.nbytes/1000000.)
    population, where_well_seq, altered = reduce_to_minimum(population)
    del altered
    t3 = time.time()
    logging_fh.write("Altering population took %f seconds\n" % (t3-t2))
    logging_fh.write("Altered population size is" + str(population.shape)+"\n")
    #spread randomly chosen population
    print "Spreading population, starting num MB: %f" % (population.nbytes/1000000.)
    population, matching = spread_population(population, space_efficient)
    t4 = time.time()
    logging_fh.write("Spreading population took %f seconds\n" % (t4-t3))
    logging_fh.write("Spread population size is" + str(population.shape)+"\n")
    #run 2-component PCA on spread population
    print "Running PCA, starting num MB: %f" % (population.nbytes/1000000.)
    pca, population = run_PCA(2, population)
    t5 = time.time()
    logging_fh.write("2-component PCA took %f seconds\n" % (t5-t4))
    two_d_plot(population, colors, png_fh=png_fh[0])
    intense_colors = make_colors(subjects, 'ethnicity', callset_names, pop_human_readable)
    logging_fh.write("Intense color mapping: %s\n" % (intense_colors))
    two_d_plot(population, intense_colors, png_fh=png_fh[1])
    return  where_well_seq, matching, pca
