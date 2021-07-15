# Pure Python + Numpy implementation of CASA's getdminfo() and getdesc()
# functions for reading metadata about .image files.

import os
import struct
from io import BytesIO
from collections import OrderedDict
from textwrap import indent

import numpy as np

__all__ = ['getdminfo', 'getdesc']

TYPES = ['bool', 'char', 'uchar', 'short', 'ushort', 'int', 'uint', 'float',
         'double', 'complex', 'dcomplex', 'string', 'table', 'arraybool',
         'arraychar', 'arrayuchar', 'arrayshort', 'arrayushort', 'arrayint',
         'arrayuint', 'arrayfloat', 'arraydouble', 'arraycomplex',
         'arraydcomplex', 'arraystr', 'record', 'other']


class AutoRepr:
    def __repr__(self):
        from pprint import pformat
        return f'{self.__class__.__name__}' + pformat(self.__dict__)


class EndianAwareFileHandle:

    def __init__(self, file_handle, endian, original_filename):
        self.file_handle = file_handle
        self.endian = endian
        self.original_filename = original_filename

    def read(self, n=None):
        return self.file_handle.read(n)

    def tell(self):
        return self.file_handle.tell()

    def seek(self, n):
        return self.file_handle.seek(n)


def with_nbytes_prefix(func):
    def wrapper(*args):
        if hasattr(args[0], 'tell'):
            self = None
            f = args[0]
            args = args[1:]
        else:
            self = args[0]
            f = args[1]
            args = args[2:]
        start = f.tell()
        nbytes = int(read_int32(f))
        print('-> calling {0} with {1} bytes starting at {2}'.format(func, nbytes, start))
        if nbytes == 0:
            return
        b = EndianAwareFileHandle(BytesIO(f.read(nbytes - 4)), f.endian, f.original_filename)
        if self:
            result = func(self, b, *args)
        else:
            result = func(b, *args)
        end = f.tell()
        print('-> ended {0} at {1}'.format(func, end))
        if end - start != nbytes:
            raise IOError('Function {0} read {1} bytes instead of {2}'
                          .format(func, end - start, nbytes))
        return result
    return wrapper


def read_bool(f):
    return f.read(1) == b'\x01'


def read_int32(f):
    return np.int32(struct.unpack(f.endian + 'i', f.read(4))[0])


def read_int64(f):
    return np.int64(struct.unpack(f.endian + 'q', f.read(8))[0])


def read_float32(f):
    return np.float32(struct.unpack(f.endian + 'f', f.read(4))[0])


def read_float64(f):
    return np.float64(struct.unpack(f.endian + 'd', f.read(8))[0])


def read_complex64(f):
    return np.complex64(read_float32(f) + 1j * read_float32(f))


def read_complex128(f):
    return np.complex128(read_float64(f) + 1j * read_float64(f))


def read_string(f):
    value = read_int32(f)
    print('value', value)
    return f.read(int(value)).replace(b'\x00', b'').decode('ascii')


@with_nbytes_prefix
def read_iposition(f):

    stype, sversion = read_type(f)

    if stype != 'IPosition' or sversion != 1:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    nelem = read_int32(f)

    return np.array([read_int32(f) for i in range(nelem)], dtype=int)


ARRAY_ITEM_READERS = {
    'float': ('float', read_float32, np.float32),
    'double': ('double', read_float64, np.float64),
    'dcomplex': ('void', read_complex128, np.complex128),
    'string': ('String', read_string, '<U16'),
    'int': ('Int', read_int32, int)
}


@with_nbytes_prefix
def read_array(f, arraytype):

    typerepr, reader, dtype = ARRAY_ITEM_READERS[arraytype]

    stype, sversion = read_type(f)

    if stype != f'Array<{typerepr}>' or sversion != 3:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    ndim = read_int32(f)
    shape = [read_int32(f) for i in range(ndim)]
    size = read_int32(f)

    values = [reader(f) for i in range(size)]

    return np.array(values, dtype=dtype).reshape(shape)


def read_type(f):
    tp = read_string(f)
    version = read_int32(f)
    return tp, version


@with_nbytes_prefix
def read_record(f):

    stype, sversion = read_type(f)

    if stype != 'Record' or sversion != 1:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    read_record_desc(f)

    # Not sure what the following value is
    read_int32(f)


@with_nbytes_prefix
def read_record_desc(f):

    stype, sversion = read_type(f)

    if stype != 'RecordDesc' or sversion != 2:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    # Not sure what the following value is
    nrec = read_int32(f)

    records = OrderedDict()

    for i in range(nrec):
        name = read_string(f)
        rectype = TYPES[read_int32(f)]
        records[name] = {'type': rectype}
        # Here we don't actually load in the data for may of the types - hence
        # why we don't do anything with the values we read in.
        if rectype in ('bool', 'int', 'uint', 'float', 'double',
                       'complex', 'dcomplex', 'string'):
            f.read(4)
        elif rectype == 'table':
            f.read(8)
        elif rectype.startswith('array'):
            read_iposition(f)
            f.read(4)
        elif rectype == 'record':
            read_record_desc(f)
            read_int32(f)
        else:
            raise NotImplementedError("Support for type {0} in RecordDesc not implemented".format(rectype))

    return records


@with_nbytes_prefix
def read_table_record(f):

    stype, sversion = read_type(f)

    if stype != 'TableRecord' or sversion != 1:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    records = read_record_desc(f)

    print(records)

    unknown = read_int32(f)  # noqa

    for name, values in records.items():
        rectype = values['type']
        if rectype == 'bool':
            records[name] = read_bool(f)
        elif rectype == 'int':
            records[name] = int(read_int32(f))
        elif rectype == 'uint':
            records[name] = int(read_int32(f))
        elif rectype == 'float':
            records[name] = float(read_float32(f))
        elif rectype == 'double':
            records[name] = float(read_float64(f))
        elif rectype == 'complex':
            records[name] = complex(read_complex64(f))
        elif rectype == 'dcomplex':
            records[name] = complex(read_complex128(f))
        elif rectype == 'string':
            records[name] = read_string(f)
        elif rectype == 'table':
            records[name] = 'Table: ' + os.path.abspath(os.path.join(f.original_filename, read_string(f)))
        elif rectype == 'arrayint':
            records[name] = read_array(f, 'int')
        elif rectype == 'arrayfloat':
            records[name] = read_array(f, 'float')
        elif rectype == 'arraydouble':
            records[name] = read_array(f, 'double')
        elif rectype == 'arraycomplex':
            records[name] = read_array(f, 'complex')
        elif rectype == 'arraydcomplex':
            records[name] = read_array(f, 'dcomplex')
        elif rectype == 'arraystr':
            records[name] = read_array(f, 'string')
        elif rectype == 'record':
            records[name] = read_table_record(f)
        else:
            raise NotImplementedError("Support for type {0} in TableRecord not implemented".format(rectype))

    return dict(records)


def check_type_and_version(f, name, versions):
    if np.isscalar(versions):
        versions = [versions]
    stype, sversion = read_type(f)
    if stype != name or sversion not in versions:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))
    return sversion


class Table(AutoRepr):

    @classmethod
    @with_nbytes_prefix
    def read(cls, f):

        self = cls()

        version = check_type_and_version(f, 'Table', 2)

        self.nrow = read_int32(f)
        self.fmt = read_int32(f)  # noqa
        self.name = read_string(f)  # noqa

        # big_endian = fmt == 0  # noqa

        self.description = TableDesc.read(f, self.nrow)

        self.column_set = ColumnSet.read(f, ncol=self.description.ncol)

        return self


class TableDesc(AutoRepr):

    @classmethod
    @with_nbytes_prefix
    def read(cls, f, nrow):

        self = cls()

        check_type_and_version(f, 'TableDesc', 2)

        unknown1 = read_int32(f)  # noqa
        unknown2 = read_int32(f)  # noqa
        unknown3 = read_string(f)  # noqa

        self.keywords = read_table_record(f)
        self.private_keywords = read_table_record(f)

        self.ncol = read_int32(f)

        self.column_description = []

        for icol in range(self.ncol):
            if icol > 0:
                read_int32(f)
            self.column_description.append(read_column_desc(f))

        return self

    # def as_casa_dict(self):

    #             desc['_keywords_'] = read_table_record(f)
    #     desc['_define_hypercolumn_'] = {}

    #     hypercolumn = read_table_record(f)
    #     desc['_private_keywords_'] = hypercolumn
    #     if hypercolumn:
    #         name = list(hypercolumn)[0].split('_')[1]
    #         value = list(hypercolumn.values())[0]
    #         desc['_define_hypercolumn_'][name] = {'HCcoordnames': value['coord'],
    #                                             'HCdatanames': value['data'],
    #                                             'HCidnames': value['id'],
    #                                             'HCndim': value['ndim']}

    #     ncol = read_int32(f)


class StandardStMan(AutoRepr):

    @classmethod
    @with_nbytes_prefix
    def read(cls, f):
        self = cls()
        check_type_and_version(f, 'SSM', 2)
        self.name = read_string(f)
        self.column_offset = Block.read(f, read_int32)
        self.column_index_map = Block.read(f, read_int32)
        return self

class Block(AutoRepr):

    @classmethod
    def read(cls, f, func):
        self = cls()
        self.nr = read_int32(f)
        self.name = read_string(f)
        self.version = read_int32(f)
        self.size = read_int32(f)
        self.elements = [func(f) for i in range(self.size)]
        return self


class ColumnSet(AutoRepr):

    @classmethod
    def read(cls, f, ncol):

        self = cls()

        version = read_int32(f)  # can be negative
        # See full logic in ColumnSet.getFile
        version = -version

        if version != 2:
            raise NotImplementedError('Support for ColumnSet version {0} not implemented'.format(version))

        self.nrow = read_int32(f)
        self.nrman = read_int32(f)
        self.nr = read_int32(f)

        # Construct data managers

        data_manager_cls = []

        for i in range(self.nr):

            name = read_string(f)
            seqnr = read_int32(f)
            print(name, seqnr)

            if name == 'StandardStMan':
                dm_cls = StandardStMan
            else:
                raise NotImplementedError('Data manager {0} not supported'.format(name))

            data_manager_cls.append(dm_cls)

        self.columns = [PlainColumn.read(f) for index in range(ncol)]

        # Prepare data managers

        f.read(8)  # includes a length in bytes and bebebebe, need to check how this behaves when multiple DMs are present

        self.data_managers = []

        for i in range(self.nr):

            self.data_managers.append(data_manager_cls[i].read(f))

        return self


class PlainColumn(AutoRepr):

    @classmethod
    def read(cls, f):

        self = cls()

        version = read_int32(f)

        if version < 2:
            raise NotImplementedError('Support for PlainColumn version {0} not implemented'.format(version))

        name = read_string(f)

        self.data = ScalarColumnData.read(f)

        return self


class ScalarColumnData(AutoRepr):

    @classmethod
    def read(cls, f):

        self = cls()

        version = read_int32(f)
        self.seqnr = read_int32(f)

        return self



def read_column_desc(f):

    unknown = read_int32(f)  # noqa

    stype, sversion = read_type(f)

    if not stype.startswith(('ScalarColumnDesc', 'ArrayColumnDesc')) or sversion != 1:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    desc = {}
    name = read_string(f)
    print('NAME', name)
    desc['comment'] = read_string(f)
    desc['dataManagerType'] = read_string(f).replace('Shape', 'Cell')
    desc['dataManagerGroup'] = read_string(f)
    if desc['dataManagerGroup'] == 'StandardStMan':
        desc['dataManagerGroup'] = 'SSM'
    desc['valueType'] = TYPES[read_int32(f)]
    desc['option'] = read_int32(f)
    ndim = read_int32(f)
    if ndim > 0:
        ipos = read_iposition(f)  # noqa
        desc['ndim'] = ndim
    desc['maxlen'] = read_int32(f)
    desc['keywords'] = read_table_record(f)
    print('HERE')
    if desc['valueType'] in ('ushort', 'short'):
        print(repr(f.read(2)))
    if desc['valueType'] in ('uint', 'int', 'float', 'string'):
        print(repr(f.read(4)))
    elif desc['valueType'] in ('double', 'complex'):
        print(repr(f.read(8)))
    elif desc['valueType'] in ('dcomplex'):
        print(repr(f.read(16)))
    return {name: desc}


@with_nbytes_prefix
def read_tiled_st_man(f):

    # The code in this function corresponds to TiledStMan::headerFileGet
    # https://github.com/casacore/casacore/blob/75b358be47039250e03e5042210cbc60beaaf6e4/tables/DataMan/TiledStMan.cc#L1086

    stype, sversion = read_type(f)

    if stype != 'TiledStMan' or sversion != 2:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    st_man = {}
    st_man['SPEC'] = {}

    st_man['BIGENDIAN'] = f.read(1) == b'\x01'  # noqa

    seqnr = read_int32(f)
    if seqnr != 0:
        raise ValueError("Expected seqnr to be 0, got {0}".format(seqnr))
    st_man['SEQNR'] = seqnr
    st_man['SPEC']['SEQNR'] = seqnr

    nrows = read_int32(f)
    if nrows != 1:
        raise ValueError("Expected nrows to be 1, got {0}".format(nrows))

    ncols = read_int32(f)
    if ncols != 1:
        raise ValueError("Expected ncols to be 1, got {0}".format(ncols))

    dtype = read_int32(f)  # noqa

    column_name = read_string(f)
    st_man['COLUMNS'] = np.array([column_name], dtype='<U16')
    st_man['NAME'] = column_name

    max_cache_size = read_int32(f)
    st_man['SPEC']['MAXIMUMCACHESIZE'] = max_cache_size
    st_man['SPEC']['MaxCacheSize'] = max_cache_size

    ndim = read_int32(f)

    nrfile = read_int32(f)  # 1
    if nrfile != 1:
        raise ValueError("Expected nrfile to be 1, got {0}".format(nrfile))

    # The following flag seems to control whether or not the TSM file is
    # opened by CASA, and is probably safe to ignore here.
    flag = bool(f.read(1))

    # The following two values are unknown, but are likely relevant when there
    # are more that one field in the image.

    mode = read_int32(f)
    unknown = read_int32(f)  # 0

    bucket = st_man['SPEC']['HYPERCUBES'] = {}
    bucket = st_man['SPEC']['HYPERCUBES']['*1'] = {}

    if mode == 1:
        total_cube_size = read_int32(f)
    elif mode == 2:
        total_cube_size = read_int64(f)
    else:
        raise ValueError('Unexpected value {0} at position {1}'.format(mode, f.tell() - 8))

    unknown = read_int32(f)  # 1
    unknown = read_int32(f)  # 1

    read_record(f)

    flag = f.read(1)  # noqa

    ndim = read_int32(f)  # noqa

    bucket['CubeShape'] = bucket['CellShape'] = read_iposition(f)
    bucket['TileShape'] = read_iposition(f)
    bucket['ID'] = {}
    bucket['BucketSize'] = int(total_cube_size /
                               np.product(np.ceil(bucket['CubeShape'] / bucket['TileShape'])))

    unknown = read_int32(f)  # noqa
    unknown = read_int32(f)  # noqa

    st_man['TYPE'] = 'TiledCellStMan'

    return st_man


@with_nbytes_prefix
def read_dminfo(f):

    stype, sversion = read_type(f)

    if stype == 'TiledCellStMan' and sversion == 1:
        return read_tiled_cell_st_man(f)
    elif stype == 'StandardStMan' and sversion == 3:
        return read_standard_st_man(f)
    else:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))


def read_tiled_cell_st_man(f):

    default_tile_shape = read_iposition(f)

    st_man = read_tiled_st_man(f)

    st_man['SPEC']['DEFAULTTILESHAPE'] = default_tile_shape

    return {'*1': st_man}


def read_standard_st_man(f):

    # pos = f.tell()
    # print(repr(f.read(10000)))
    # f.seek(pos)

    # SSMBase::readHeader()
    # https://github.com/casacore/casacore/blob/d6da19830fa470bdd8434fd855abe79037fda78c/tables/DataMan/SSMBase.cc#L415

    big_endian = f.read(1) == b'\x01'  # noqa

    bucket_size = read_int32(f)
    number_of_buckets = read_int32(f)
    persistent_cache = read_int32(f)
    number_of_free_buckets = read_int32(f)
    first_free_bucket = read_int32(f)
    number_of_bucket_for_index = read_int32(f)
    first_index_bucket_number = read_int32(f)
    idx_bucket_offset = read_int32(f)
    last_string_bucket = read_int32(f)
    index_length = read_int32(f)
    number_indices = read_int32(f)

    print(bucket_size)
    print(number_of_buckets)
    print(persistent_cache)
    print(number_of_free_buckets)
    print(first_free_bucket)
    print(number_of_bucket_for_index)
    print(first_index_bucket_number)
    print(idx_bucket_offset)
    print(last_string_bucket)
    print(index_length)
    print(number_indices)

    # st_man = read_tiled_st_man(f)

    st_man = {}
    st_man['SPEC'] = {}
    st_man['SPEC']['BUCKETSIZE'] = bucket_size
    st_man['SPEC']['IndexLength'] = index_length
    st_man['SPEC']['MaxCacheSize'] = persistent_cache  # NOTE: not sure if correct
    st_man['SPEC']['PERSCACHESIZE'] = persistent_cache

    st_man['TYPE'] = 'StandardStMan'
    st_man['NAME'] = 'SSM'
    st_man['SEQNR'] = 0

    print(st_man)

    return {'*1': st_man}

    # Bucket 0 contains SSMIndex:
    # https://github.com/casacore/casacore/blob/d6da19830fa470bdd8434fd855abe79037fda78c/tables/DataMan/SSMIndex.cc#L55

    # Then it contains another index at offset given by idx_bucket_offset

def getdminfo(filename, endian='>'):
    """
    Return the same output as CASA's getdminfo() function, namely a dictionary
    with metadata about the .image file, parsed from the ``table.f0`` file.
    """

    with open(os.path.join(filename, 'table.f0'), 'rb') as f_orig:

        f = EndianAwareFileHandle(f_orig, endian, filename)

        magic = f.read(4)
        if magic != b'\xbe\xbe\xbe\xbe':
            raise ValueError('Incorrect magic code: {0}'.format(magic))

        dminfo = read_dminfo(f)

    print(dminfo['*1']['TYPE'])

    # if dminfo['*1']['TYPE'] == 'StandardStMan':
    #     desc = getdesc(filename)
    #     print(desc)
    #     dminfo['*1']['COLUMNS'] = []
    #     for key in desc:
    #         print(key)
    #         if 'dataManagerGroup' in desc[key]:
    #             dminfo['*1']['COLUMNS'].append(key)

    #     dminfo['*1']['COLUMNS'] = np.array(sorted(dminfo['*1']['COLUMNS']), dtype='<U16')

    return dminfo


def getdesc(filename, endian='>'):
    """
    Return the same output as CASA's getdesc() function, namely a dictionary
    with metadata about the .image file, parsed from the ``table.dat`` file.
    """

    print(filename)

    with open(os.path.join(filename, 'table.dat'), 'rb') as f_orig:

        f = EndianAwareFileHandle(f_orig, endian, filename)

        magic = f.read(4)
        if magic != b'\xbe\xbe\xbe\xbe':
            raise ValueError('Incorrect magic code: {0}'.format(magic))

        result = Table.read(f)
        return result


# New OO interface which will give more flexibility
