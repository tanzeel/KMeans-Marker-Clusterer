import pandas as pd
from sqlalchemy import create_engine

def clean_database():
    # Create engine
    engine = create_engine('sqlite:///../mcluster/db.sqlite3')

    with engine.connect() as con:
        con.execute("DELETE FROM mcluster_sale")

def query_database():
    # Create engine
    engine = create_engine('sqlite:///../mcluster/db.sqlite3')

    with engine.connect() as con:
        sales = con.execute("SELECT COUNT(*) FROM mcluster_sale;").fetchall()

    print(sales)


def migrate_dataframe(df, method):
    # Create engine
    engine = create_engine('sqlite:///../mcluster/db.sqlite3', echo=False)

    df['sale_date'] = pd.to_datetime(df['sale_date'], dayfirst=True)

    df.to_sql('mcluster_sale', engine, if_exists=method, index_label='id')
    # Use if_exists='append' when adding new data

def main():
    gdf = pd.read_csv('./data.csv')

    gdf['ed'] = gdf['ed'].fillna('None')

    gdf.drop_duplicates(subset=['sale_date', 'address', 'postcode', 'county', 'price', 'nfma',
       'vat_ex', 'DoP', 'PSD', 'region', 'latitude', 'longitude', 'ed'])

    migrate_dataframe(gdf, 'replace')
    query_database()


if __name__ == "__main__":
    main()