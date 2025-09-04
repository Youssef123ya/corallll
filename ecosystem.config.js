module.exports = {
  apps: [{
    name: 'coral-reef-guardian',
    script: 'uvicorn',
    args: 'app:app --host 0.0.0.0 --port 8000 --workers 4',
    cwd: '/home/user/webapp',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      NODE_ENV: 'production',
      PYTHONPATH: '/home/user/webapp',
      HOST: '0.0.0.0',
      PORT: '8000',
      DEBUG: 'false',
      LOG_LEVEL: 'info'
    },
    env_production: {
      NODE_ENV: 'production',
      PYTHONPATH: '/home/user/webapp',
      HOST: '0.0.0.0',
      PORT: '8000',
      DEBUG: 'false',
      LOG_LEVEL: 'warning'
    },
    env_development: {
      NODE_ENV: 'development',
      PYTHONPATH: '/home/user/webapp',
      HOST: '0.0.0.0',
      PORT: '8000',
      DEBUG: 'true',
      LOG_LEVEL: 'debug'
    },
    error_file: './logs/coral-app-error.log',
    out_file: './logs/coral-app-out.log',
    log_file: './logs/coral-app-combined.log',
    time: true,
    merge_logs: true,
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
  }]
};