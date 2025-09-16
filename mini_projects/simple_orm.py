"""
Simple ORM Mini Project
=======================
A lightweight ORM implementation using descriptors and metaclasses
to demonstrate advanced Python features.
"""

import sqlite3
from typing import Any, Dict, List, Optional, Type, get_type_hints
from abc import ABC, abstractmethod


class Field:
    """Base field descriptor for ORM models."""
    
    def __init__(self, field_type: type, primary_key: bool = False, 
                 nullable: bool = True, default: Any = None):
        self.field_type = field_type
        self.primary_key = primary_key
        self.nullable = nullable
        self.default = default
        self.name = None  # Set by metaclass
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name, self.default)
    
    def __set__(self, instance, value):
        if value is None and not self.nullable and not self.primary_key:
            raise ValueError(f"Field {self.name} cannot be None")
        
        if value is not None and not isinstance(value, self.field_type):
            try:
                value = self.field_type(value)
            except (ValueError, TypeError):
                raise TypeError(f"Field {self.name} must be of type {self.field_type.__name__}")
        
        instance.__dict__[self.name] = value
    
    def to_sql_type(self) -> str:
        """Convert Python type to SQL type."""
        type_mapping = {
            int: 'INTEGER',
            str: 'TEXT',
            float: 'REAL',
            bool: 'INTEGER',  # SQLite stores booleans as integers
        }
        return type_mapping.get(self.field_type, 'TEXT')


class CharField(Field):
    def __init__(self, max_length: int = 255, **kwargs):
        super().__init__(str, **kwargs)
        self.max_length = max_length
    
    def __set__(self, instance, value):
        if value and len(str(value)) > self.max_length:
            raise ValueError(f"Field {self.name} exceeds max length of {self.max_length}")
        super().__set__(instance, value)


class IntegerField(Field):
    def __init__(self, **kwargs):
        super().__init__(int, **kwargs)


class FloatField(Field):
    def __init__(self, **kwargs):
        super().__init__(float, **kwargs)


class BooleanField(Field):
    def __init__(self, **kwargs):
        super().__init__(bool, **kwargs)


class ModelMeta(type):
    """Metaclass for ORM models."""
    
    def __new__(cls, name, bases, attrs):
        # Don't process the base Model class
        if name == 'Model':
            return super().__new__(cls, name, bases, attrs)
        
        # Collect fields
        fields = {}
        for key, value in attrs.items():
            if isinstance(value, Field):
                fields[key] = value
        
        # Store fields metadata
        attrs['_fields'] = fields
        attrs['_table_name'] = attrs.get('_table_name', name.lower())
        
        # Find primary key
        primary_keys = [name for name, field in fields.items() if field.primary_key]
        if len(primary_keys) > 1:
            raise ValueError(f"Model {name} has multiple primary keys")
        attrs['_primary_key'] = primary_keys[0] if primary_keys else None
        
        return super().__new__(cls, name, bases, attrs)


class QuerySet:
    """Simple query builder and executor."""
    
    def __init__(self, model_class: Type['Model'], db_connection):
        self.model_class = model_class
        self.db = db_connection
        self._where_clauses = []
        self._order_by = []
        self._limit = None
    
    def filter(self, **kwargs):
        """Add WHERE clauses."""
        new_qs = QuerySet(self.model_class, self.db)
        new_qs._where_clauses = self._where_clauses.copy()
        new_qs._order_by = self._order_by.copy()
        new_qs._limit = self._limit
        
        for field, value in kwargs.items():
            if field not in self.model_class._fields:
                raise ValueError(f"Unknown field: {field}")
            new_qs._where_clauses.append((field, value))
        
        return new_qs
    
    def order_by(self, *fields):
        """Add ORDER BY clauses."""
        new_qs = QuerySet(self.model_class, self.db)
        new_qs._where_clauses = self._where_clauses.copy()
        new_qs._order_by = list(fields)
        new_qs._limit = self._limit
        return new_qs
    
    def limit(self, count: int):
        """Add LIMIT clause."""
        new_qs = QuerySet(self.model_class, self.db)
        new_qs._where_clauses = self._where_clauses.copy()
        new_qs._order_by = self._order_by.copy()
        new_qs._limit = count
        return new_qs
    
    def _build_query(self) -> tuple:
        """Build SQL query and parameters."""
        table_name = self.model_class._table_name
        query = f"SELECT * FROM {table_name}"
        params = []
        
        if self._where_clauses:
            where_parts = []
            for field, value in self._where_clauses:
                where_parts.append(f"{field} = ?")
                params.append(value)
            query += " WHERE " + " AND ".join(where_parts)
        
        if self._order_by:
            query += " ORDER BY " + ", ".join(self._order_by)
        
        if self._limit:
            query += f" LIMIT {self._limit}"
        
        return query, params
    
    def all(self) -> List['Model']:
        """Execute query and return all results."""
        query, params = self._build_query()
        cursor = self.db.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            instance = self.model_class()
            for i, field_name in enumerate(self.model_class._fields.keys()):
                setattr(instance, field_name, row[i])
            results.append(instance)
        
        return results
    
    def first(self) -> Optional['Model']:
        """Return first result or None."""
        results = self.limit(1).all()
        return results[0] if results else None
    
    def get(self, **kwargs) -> 'Model':
        """Get single object or raise exception."""
        results = self.filter(**kwargs).all()
        if not results:
            raise ValueError("Object not found")
        if len(results) > 1:
            raise ValueError("Multiple objects found")
        return results[0]


class Model(metaclass=ModelMeta):
    """Base ORM model class."""
    
    _db_connection = None
    
    @classmethod
    def set_db_connection(cls, connection):
        """Set database connection for all models."""
        cls._db_connection = connection
    
    @classmethod
    def create_table(cls):
        """Create table for this model."""
        if not cls._db_connection:
            raise RuntimeError("No database connection set")
        
        fields_sql = []
        for name, field in cls._fields.items():
            sql_type = field.to_sql_type()
            constraints = []
            
            if field.primary_key:
                constraints.append("PRIMARY KEY")
            if not field.nullable and not field.primary_key:
                constraints.append("NOT NULL")
            
            field_sql = f"{name} {sql_type}"
            if constraints:
                field_sql += " " + " ".join(constraints)
            fields_sql.append(field_sql)
        
        query = f"CREATE TABLE IF NOT EXISTS {cls._table_name} ({', '.join(fields_sql)})"
        cls._db_connection.execute(query)
        cls._db_connection.commit()
    
    @classmethod
    def objects(cls) -> QuerySet:
        """Return QuerySet for this model."""
        if not cls._db_connection:
            raise RuntimeError("No database connection set")
        return QuerySet(cls, cls._db_connection)
    
    def save(self):
        """Save instance to database."""
        if not self._db_connection:
            raise RuntimeError("No database connection set")
        
        field_names = list(self._fields.keys())
        field_values = [getattr(self, name) for name in field_names]
        
        # Check if this is an update (has primary key value) or insert
        if self._primary_key and getattr(self, self._primary_key) is not None:
            # Update existing record
            set_clauses = [f"{name} = ?" for name in field_names if name != self._primary_key]
            update_values = [getattr(self, name) for name in field_names if name != self._primary_key]
            update_values.append(getattr(self, self._primary_key))
            
            query = f"UPDATE {self._table_name} SET {', '.join(set_clauses)} WHERE {self._primary_key} = ?"
            self._db_connection.execute(query, update_values)
        else:
            # Insert new record
            placeholders = ", ".join(["?" for _ in field_names])
            query = f"INSERT INTO {self._table_name} ({', '.join(field_names)}) VALUES ({placeholders})"
            cursor = self._db_connection.execute(query, field_values)
            
            # Set primary key for new records
            if self._primary_key and getattr(self, self._primary_key) is None:
                setattr(self, self._primary_key, cursor.lastrowid)
        
        self._db_connection.commit()
    
    def delete(self):
        """Delete instance from database."""
        if not self._db_connection:
            raise RuntimeError("No database connection set")
        
        if not self._primary_key:
            raise ValueError("Cannot delete object without primary key")
        
        pk_value = getattr(self, self._primary_key)
        if pk_value is None:
            raise ValueError("Cannot delete object with null primary key")
        
        query = f"DELETE FROM {self._table_name} WHERE {self._primary_key} = ?"
        self._db_connection.execute(query, (pk_value,))
        self._db_connection.commit()
    
    def __repr__(self):
        field_strs = []
        for name in self._fields.keys():
            value = getattr(self, name)
            field_strs.append(f"{name}={value!r}")
        return f"{self.__class__.__name__}({', '.join(field_strs)})"


# Example models
class User(Model):
    _table_name = 'users'
    
    id = IntegerField(primary_key=True)
    username = CharField(max_length=50, nullable=False)
    email = CharField(max_length=100, nullable=False)
    age = IntegerField(nullable=True)
    is_active = BooleanField(default=True)


class Post(Model):
    _table_name = 'posts'
    
    id = IntegerField(primary_key=True)
    title = CharField(max_length=200, nullable=False)
    content = CharField(max_length=1000)
    user_id = IntegerField(nullable=False)  # Simple foreign key
    views = IntegerField(default=0)


def demo_orm():
    """Demonstrate ORM functionality."""
    print("🗄️  Simple ORM Demo")
    
    # Setup database
    conn = sqlite3.connect(':memory:')  # In-memory database for demo
    Model.set_db_connection(conn)
    
    # Create tables
    User.create_table()
    Post.create_table()
    print("✅ Tables created")
    
    # Create users
    user1 = User()
    user1.username = "alice"
    user1.email = "alice@example.com"
    user1.age = 25
    user1.save()
    
    user2 = User()
    user2.username = "bob"
    user2.email = "bob@example.com"
    user2.age = 30
    user2.is_active = False
    user2.save()
    
    print(f"✅ Created users: {user1}, {user2}")
    
    # Create posts
    post1 = Post()
    post1.title = "Hello World"
    post1.content = "This is my first post!"
    post1.user_id = user1.id
    post1.views = 5
    post1.save()
    
    post2 = Post()
    post2.title = "Python ORM"
    post2.content = "Building an ORM with metaclasses is fun!"
    post2.user_id = user1.id
    post2.views = 15
    post2.save()
    
    print(f"✅ Created posts: {post1}, {post2}")
    
    # Query examples
    print("\n📊 Query Examples:")
    
    # Get all users
    all_users = User.objects().all()
    print(f"All users: {all_users}")
    
    # Filter users
    active_users = User.objects().filter(is_active=True).all()
    print(f"Active users: {active_users}")
    
    # Get single user
    alice = User.objects().get(username="alice")
    print(f"Found Alice: {alice}")
    
    # Order and limit
    popular_posts = Post.objects().order_by("-views").limit(1).all()
    print(f"Most popular post: {popular_posts}")
    
    # Update
    alice.age = 26
    alice.save()
    print(f"Updated Alice: {User.objects().get(username='alice')}")
    
    # Delete
    post1.delete()
    remaining_posts = Post.objects().all()
    print(f"Posts after deletion: {remaining_posts}")
    
    conn.close()
    print("\n✅ ORM demo completed!")


if __name__ == "__main__":
    demo_orm()