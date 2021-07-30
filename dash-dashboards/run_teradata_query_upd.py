import argparse
import os
import datetime
import teradatasql
import tqdm
import gzip
import io

def getQuery(fname):
  f = open(fname, 'r')
  query = ' '.join([x[:x.find('--')].strip() for x in f.readlines() if not x.strip().startswith('--')])
  f.close()
  return query

def cleanstr(x):
  return '' if x is None else (str(x) if type(x) is not str else (x.rstrip(' ') if x.find(',') < 0 else '"'+x.rstrip(' ')+'"'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_filename',
                        dest='output_filename', action='store', type=str,
                        help='Output csv', default='output.csv')
    parser.add_argument('-q', '--query_filename',
                        dest='queryfname', action='store', type=str,
                        help='query_filename', required=True)
    args = parser.parse_args()
    
    print("Opening connection to teradata at ", datetime.datetime.now())
    con=teradatasql.connect(
      host='edwprod.dw.medcity.net',
      user=os.getenv('TERADATA_MD_LOGON_USER'),
      password=os.getenv('TERADATA_MD_LOGON_PASSWORD'),
      teradata_values="false")
    cur=con.cursor()
    
    print("Connection opened at ", datetime.datetime.now())
    query = getQuery(args.queryfname)
    print("Read query at ", datetime.datetime.now())
    
    cur.execute(query)
    print("Executed query at ", datetime.datetime.now())
    print(cur.rowcount,' rows returned')
    prog=tqdm.tqdm(total=cur.rowcount,desc=' rows parsed & written to '+args.output_filename)
    
    chunksz=10000
    f=open(args.output_filename,'w') if not args.output_filename.endswith('.gz') else io.TextIOWrapper(gzip.open(args.output_filename,'w'))
    # write the header
    f.write('|'.join([cleanstr(cur.description[coln][0]) for coln in range(len(cur.description))]) + '\n')
    currow=0
    totalc=cur.rowcount
    while currow <= totalc:
      v=['|'.join([cleanstr(x) for x in y]) for y in cur.fetchmany(chunksz)]
      f.write('\n'.join(v) + '\n')
      currow+=chunksz
      prog.update(n=len(v))
    prog.close()
    f.close()
    cur.close()
    con.close()
    print("Wrote data at ", datetime.datetime.now())
