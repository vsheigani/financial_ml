import numpy as np

def _embed(time_series: np.ndarray, order: int = 3, embedding_delay: int = 1) -> np.ndarray:
    """Embed the time series into a matrix of shape (embedding_order, n_times - (embedding_order - 1) * embedding_delay)

    Parameters
    ----------
    time_series : 1d-array numpy array
        Time series
    order : int
        Embedding order
    embedding_delay : int
        Time delay between samples

    Returns
    -------
    embedded : np.ndarray, shape (n_times - (order - 1) * embedding_delay, order)
        Embedded time-series.
    """
    num_samples = len(time_series)
    embedded_matrix = np.empty((order, num_samples - (order - 1) * embedding_delay))
    for order_index in range(order):
        embedded_matrix[order_index] = time_series[order_index * embedding_delay:order_index * embedding_delay + embedded_matrix.shape[1]]
    return embedded_matrix.T



def shannon_entropy(time_series: np.ndarray) -> float:
    """ Calculate the Shannon entropy of the time series.
    In information theory, the entropy of a random 
    variable quantifies the average level of information
    associated with the variable's probability distribution.
    
    The Shannon entropy is defined as: 
    H(X) = -sum(p(x) * log2(p(x)))
    where p(x) is the probability of the event x.

    Args:
        time_series: Vector or string of the sample data

    Returns:
        The Shannon Entropy as float value
    """

    # Check if string
    if not isinstance(time_series, str):
        time_series = list(time_series)

    # Create a frequency data
    unique_elements = list(set(time_series))
    probability_list = []
    for element in unique_elements:
        occurrence_count = 0.
        for value in time_series:
            if value == element:
                occurrence_count += 1
        probability_list.append(float(occurrence_count) / len(time_series))

    # Shannon entropy
    entropy = 0.0
    for probability in probability_list:
        entropy += probability * np.log2(probability)
    entropy = -entropy
    return entropy


def sample_entropy(time_series: np.ndarray, sample_length: int, tolerance: float | None = None) -> np.ndarray:
    """Calculates the sample entropy of degree 'sample_length - 1' of a time_series.

    Args:
        time_series: numpy array of time series
        sample_length: length of longest template vector
        tolerance: The maximum allowed difference between the template and the remaining time series
        (defaults to 0.1 * std(time_series)))
    Returns:
        Array of sample entropies
    """
    # assert isinstance(time_series, np.ndarray), "time_series must be a numpy array"
    # assert sample_length > 1, "sample_length must be greater than 1"
    # The parameter 'sample_length' is equal to m + 1 in Ref[1].
    embedding_degree = sample_length - 1

    if tolerance is None:
        tolerance = 0.1 * np.std(time_series)
    else:
        tolerance = float(tolerance)

    num_samples = len(time_series)

    # match_counts is a vector that holds the number of matches between the template and the remaining time series
    # match_counts[k] holds matches for templates of length k
    match_counts = np.zeros(embedding_degree + 2)
    match_counts[0] = num_samples * (num_samples - 1) / 2

    for start_index in range(num_samples - embedding_degree - 1):
        # template_vector has 'embedding_degree + 1' elements
        template_vector = time_series[start_index:(start_index + embedding_degree + 1)] 
        remaining_time_series = time_series[start_index + 1:]

        search_indices = np.arange(len(remaining_time_series) - embedding_degree, dtype=np.int32)
        for template_length in range(1, len(template_vector) + 1):
            # match_mask is a boolean array of the same length as search_indices which is True
            # for the indices where the condition is met
            match_mask = np.abs(remaining_time_series[search_indices] - template_vector[template_length - 1]) < tolerance
            match_counts[template_length] += np.sum(match_mask)
            search_indices = search_indices[match_mask] + 1

    sample_entropy = -np.log(match_counts[1:] / match_counts[:-1])
    return sample_entropy




def permutation_entropy(time_series: np.ndarray, embedding_order: int = 3, embedding_delay: int = 1, normalize: bool = False) -> float:
    """ Calculate the Permutation Entropy of the time series.
    Permutation entropy is a measure of the complexity of a time series.
    It is calculated by embedding the time series into a vector of length
    embedding_order, and then calculating the entropy of the permutations of the
    embedded vector.

    Parameters
    ----------
    time_series : np.array
        Time series (1d-array numpy array)
    order : int
        Order of the permutation entropy
    embedding_delay : int
        Time delay between samples
    normalize : bool
        If True, divide computed entropy by log(factorial(embedding_order)) to normalize the entropy

    Returns
    -------
    permutation_entropy_value : float
        Permutation Entropy of the time series

    """

    assert isinstance(time_series, np.ndarray), "time_series must be a numpy array"
    assert embedding_order > 2, "embedding_order must be greater than 2"
    assert embedding_delay > 0, "embedding_delay must be greater than 0"

    hash_multiplier = np.power(embedding_order, np.arange(embedding_order))
    # Embed time_series and sort the order of permutations
    sorted_indices = _embed(time_series, order=embedding_order, embedding_delay=embedding_delay).argsort(kind='quicksort')
    # Associate unique integer to each permutation
    permutation_hashes = (np.multiply(sorted_indices, hash_multiplier)).sum(1)

    _, permutation_counts = np.unique(permutation_hashes, return_counts=True)
    probabilities = np.where(permutation_counts.sum() != 0, permutation_counts / permutation_counts.sum(), 0)
    permutation_entropy_value = -np.multiply(probabilities, np.log2(probabilities)).sum()
    if normalize:
        permutation_entropy_value /= np.log2(np.factorial(embedding_order))
    return permutation_entropy_value
