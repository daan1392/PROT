def diff_ND_libs(zai, xs1, xs2):
    """
    Given two nuclear data libraries processed in 238 multigroup structure, return the difference between the two nuclear data libraries. If a nuclide is in one and not the other, disregard the difference. That is, only compute the difference for which there is data available
    
    Parameters
    ----------
    zais : list
        List of zais for which the difference is to be computed
    xs1 : dict
        Dictionary of cross sections for which the keys are the nuclides.
    xs2 : dict
        Second dictionairy for which keff values are available.

    Returns
    -------

    """
    if zai not in xs1 or zai not in xs2:
        print(f"ZAI {zai} not in both cross section dictionaries, skipping")
        return None
    
    if xs1[zai].empty or xs2[zai].empty:
        if xs1[zai].empty:
            print(f"ZAI {zai} is empty in xs1")
        if  xs2[zai].empty:
            print(f"ZAI {zai} is empty in xs2")
        print(f"ZAI {zai} has empty cross section data, skipping")
        return None
    
    print(zai)
    xs_diff = xs2[zai] - xs1[zai]
    return xs_diff